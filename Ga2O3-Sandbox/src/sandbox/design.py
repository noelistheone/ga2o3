"""design — the per-lab experiment-designer (the one real lever for absolute accuracy).

The proven ceiling: cross-lab ABSOLUTE prediction is capped because the state that sets absolute
values is unrecorded and lab-specific. The ONLY way past it for a given lab is for THAT lab to run
a few controlled experiments that pin its own theta. This module makes that lever optimal.

Per-lab calibration parameters (identifiable, each pinned by a distinct experiment type):
  theta = { log10 Tf_eff : effective defect freeze-in T (dominant, ~4-orders lever on n) — pinned
                           by a DOPED-donor sample,
            E_B          : grain-boundary/barrier energy on mobility — pinned by a FILM mu,
            log10 Nd_bg  : unintentional background-donor density — pinned by an UNDOPED sample. }
Design variables the lab controls per experiment: dopant, conc, film(y/n), pO2 (anneal atmosphere).
(pO2 is a controllable knob, not an unknown; a residual pO2-offset theta is a documented extension.)

  theta_fit()     — Gauss-Newton least-squares of theta on the lab's own (condition -> measured
                    n/mu/sigma) points + Gaussian prior; Laplace posterior covariance.
  fisher()        — local Fisher information of a candidate experiment for theta.
  design_battery()— greedily choose the K experiments (D- or A-optimal) that most reduce theta
                    uncertainty, starting from what the lab already has; reports the theta-std and
                    absolute-n-dex before/after so the lab sees exactly how much its bars tighten.
  predict_with_uncertainty() — propagate the theta posterior to the 9-property bars.

Genuinely-simulated contract: theta are PHYSICAL process parameters fed to the V61 solver; fitting
them is calibration, not table-lookup. Every emitted value is full_forward.forward.
"""
from __future__ import annotations
import numpy as np
from . import full_forward as ff, kroger_db

THETA_KEYS = ["log10_Tf", "E_B", "log10_Nd_bg"]
THETA0 = {"log10_Tf": np.log10(1400.0), "E_B": 0.03, "log10_Nd_bg": np.log10(1e17)}
# Gaussian priors (std in transformed units) — regularize under-determined directions.
THETA_PRIOR_STD = {"log10_Tf": 0.18, "E_B": 0.05, "log10_Nd_bg": 1.0}
FD_STEP = {"log10_Tf": 0.03, "E_B": 0.01, "log10_Nd_bg": 0.2}
OBS_KEYS = ["log10_n", "log10_mu", "log10_sigma"]
OBS_NOISE = {"log10_n": 0.15, "log10_mu": 0.10, "log10_sigma": 0.20}   # dex (Hall scatter)

# The 9 emitted properties and the design weights (per agent-#3 B6): spend experiments only on the
# movable, high-value targets. tau/dark/PDR/V_O are order-of-magnitude-capped REGARDLESS of
# calibration (trap/interface/microstructure not in the 4-D theta) -> w=0, excluded from the design
# objective. n/sigma are the high-value movable absolutes; mu already good; Eg/dEg tight (small w).
PROP_KEYS = ["hall_n_cm3", "hall_mu_cm2Vs", "sigma_S_cm", "Eg_optical_eV", "dEg_eV",
             "V_O_cm3_quench", "dark_activation_eV", "PDR_score", "tau_score"]
PROP_LOG = {"hall_n_cm3", "sigma_S_cm", "V_O_cm3_quench"}   # scored in log10 (multiplicative)
DESIGN_WEIGHTS = {"hall_n_cm3": 1.0, "sigma_S_cm": 0.7, "hall_mu_cm2Vs": 0.3,
                  "Eg_optical_eV": 0.1, "dEg_eV": 0.1,
                  "V_O_cm3_quench": 0.0, "dark_activation_eV": 0.0,
                  "PDR_score": 0.0, "tau_score": 0.0}


def _call(theta, cond):
    return dict(T_anneal=10 ** theta["log10_Tf"], pO2=cond.get("pO2", 1e-5),
                E_B=theta["E_B"], Nd_bg=10 ** theta["log10_Nd_bg"])


def observe(cond, theta, db=None):
    """Forward-simulate [log10 n, log10 mu, log10 sigma] at a condition under theta.
    undoped_cache={} short-circuits the internal undoped reference (only dEg needs it; the designer
    scores n/mu/sigma) — a ~2x speedup that does not affect the returned channels."""
    db = db or kroger_db.load()
    c = _call(theta, cond)
    r = ff.forward(cond["dopant"], cond.get("conc", 0.0), T_anneal=c["T_anneal"], pO2=c["pO2"],
                   film=cond.get("film", True), E_B_eV=c["E_B"], Nd_bg=c["Nd_bg"], db=db,
                   undoped_cache=({} if cond.get("dopant") != "undoped" else None))
    return np.array([np.log10(max(r["hall_n_cm3"], 1.0)),
                     np.log10(max(r["hall_mu_cm2Vs"], 1e-3)),
                     np.log10(max(r["sigma_S_cm"], 1e-30))])


def jacobian(cond, theta, obs_mask=None, db=None):
    db = db or kroger_db.load()
    idx = [OBS_KEYS.index(o) for o in (obs_mask or OBS_KEYS)]
    J = np.zeros((len(idx), len(THETA_KEYS)))
    for j, key in enumerate(THETA_KEYS):
        tp = dict(theta); tm = dict(theta)
        tp[key] += FD_STEP[key]; tm[key] -= FD_STEP[key]
        J[:, j] = (observe(cond, tp, db)[idx] - observe(cond, tm, db)[idx]) / (2 * FD_STEP[key])
    return J


def fisher(cond, theta, obs_mask=None, db=None):
    obs = obs_mask or OBS_KEYS
    J = jacobian(cond, theta, obs, db)
    Winv = np.diag([1.0 / OBS_NOISE[o] ** 2 for o in obs])
    return J.T @ Winv @ J


def prior_precision():
    return np.diag([1.0 / THETA_PRIOR_STD[k] ** 2 for k in THETA_KEYS])


def theta_fit(points, theta_init=None, db=None, iters=12):
    """Fit per-lab theta by Gauss-Newton on the lab's own measurements + prior.
    points: [{dopant, conc, film, pO2?, obs_mask?, meas:{log10_n, log10_mu?, log10_sigma?}}].
    Returns (theta_hat, cov). cov's large-variance eigenvector = the theta the lab has NOT pinned."""
    db = db or kroger_db.load()
    theta = dict(theta_init or THETA0)
    P0 = prior_precision()
    m0 = np.array([THETA0[k] for k in THETA_KEYS])
    JtWJ = P0.copy()
    for _ in range(iters):
        JtWr = np.zeros(len(THETA_KEYS)); JtWJ = P0.copy()
        for pt in points:
            obs = pt.get("obs_mask") or [o for o in OBS_KEYS if o in pt["meas"]]
            idx = [OBS_KEYS.index(o) for o in obs]
            pred = observe(pt, theta, db)[idx]
            meas = np.array([pt["meas"][o] for o in obs])
            J = jacobian(pt, theta, obs, db)
            Winv = np.diag([1.0 / OBS_NOISE[o] ** 2 for o in obs])
            JtWr += J.T @ Winv @ (meas - pred)
            JtWJ += J.T @ Winv @ J
        cur = np.array([theta[k] for k in THETA_KEYS])
        step = np.linalg.solve(JtWJ, JtWr - P0 @ (cur - m0))
        cur = cur + np.clip(step, -0.5, 0.5)
        for i, k in enumerate(THETA_KEYS):
            theta[k] = float(cur[i])
        theta["log10_Tf"] = float(np.clip(theta["log10_Tf"], np.log10(1073), np.log10(2000)))
        theta["E_B"] = float(np.clip(theta["E_B"], 0.0, 0.3))
        theta["log10_Nd_bg"] = float(np.clip(theta["log10_Nd_bg"], 14.0, 20.0))
    return theta, np.linalg.inv(JtWJ)


def _info_matrix(points, theta, db):
    M = prior_precision().copy()
    for pt in points:
        M += fisher(pt, theta, pt.get("obs_mask"), db)
    return M


def design_battery(K=4, theta=None, base_points=None, pool=None, criterion="D",
                   targets=None, weights=None, db=None):
    """Recommend the K next experiments that most reduce per-lab uncertainty (greedy, batched
    Kriging-Believer on the Fisher info). criterion:
      'D' = D-optimal (max log det M) — reparam-invariant, attacks the worst-conditioned directions.
      'A' = A-optimal (min trace M^-1) — average theta variance.
      'L' = prediction-weighted (min Σ_targets Σ_props w_j J_j Σ J_j^T) — spends experiments only on
            what propagates to the lab's TARGET conditions (agent-#3 primary recommendation).
    Returns ranked conditions, marginal gains, theta-std + absolute-n-dex before/after."""
    db = db or kroger_db.load()
    theta = dict(theta or THETA0)
    base_points = base_points or []
    pool = default_pool() if pool is None else pool
    if criterion == "L" and not targets:
        targets = [{"dopant": "Si", "conc": 0.005, "film": True, "pO2": 1e-5}]
    tjacs = [property_jacobian(x, theta, db) for x in (targets or [])]
    M = _info_matrix(base_points, theta, db)
    fishers = [fisher(c, theta, c.get("obs_mask"), db) for c in pool]

    def score(M_):                                   # higher = better
        if criterion == "D":
            return np.linalg.slogdet(M_)[1]
        if criterion == "A":
            return -np.trace(np.linalg.inv(M_))
        return -predictive_Phi(targets, np.linalg.inv(M_), theta, weights, db, tjacs)

    chosen, gains, remaining = [], [], set(range(len(pool)))
    for _ in range(min(K, len(pool))):
        cur = score(M)
        best, best_val = None, -np.inf
        for i in remaining:
            val = score(M + fishers[i])
            if val > best_val:
                best_val, best = val, i
        M = M + fishers[best]; chosen.append(best); gains.append(round(float(best_val - cur), 3))
        remaining.discard(best)
    std_before = np.sqrt(np.diag(np.linalg.inv(_info_matrix(base_points, theta, db))))
    std_after = np.sqrt(np.diag(np.linalg.inv(M)))
    tf_i = THETA_KEYS.index("log10_Tf")
    return {
        "criterion": criterion,
        "battery": [pool[i] for i in chosen],
        "info_gain": gains,          # per-step improvement in the chosen criterion's score
        "theta_std_before": {k: round(float(std_before[j]), 3) for j, k in enumerate(THETA_KEYS)},
        "theta_std_after": {k: round(float(std_after[j]), 3) for j, k in enumerate(THETA_KEYS)},
        "abs_n_dex_before": round(float(std_before[tf_i] * _dlogn_dlogTf(theta, db)), 2),
        "abs_n_dex_after": round(float(std_after[tf_i] * _dlogn_dlogTf(theta, db)), 2),
        "note": "abs_n_dex = the 1-sigma absolute carrier-density band implied by the freeze-in-T "
                "uncertainty at a mid-doping donor; this is what the battery tightens.",
    }


def _dlogn_dlogTf(theta, db, cond=None):
    cond = cond or {"dopant": "Si", "conc": 0.005, "film": True, "pO2": 1e-5}
    J = jacobian(cond, theta, ["log10_n"], db)
    return abs(float(J[0, THETA_KEYS.index("log10_Tf")]))


def default_pool():
    """Menu a sputter+anneal lab can run. Spans the degeneracy-breaking axes: UNDOPED (pins Nd_bg),
    donor conc series + pO2 sweep (pins Tf), film-vs-crystal (pins E_B via mu)."""
    pool = []
    for dop, conc, mask in [
        ("undoped", 0.0, ["log10_n"]),
        ("Si", 0.001, OBS_KEYS), ("Si", 0.005, OBS_KEYS), ("Si", 0.02, OBS_KEYS),
        ("Sn", 0.005, OBS_KEYS), ("Hf", 0.005, OBS_KEYS)]:
        for film in [False, True]:
            for pO2 in [1e-3, 1e-7]:
                pool.append({"dopant": dop, "conc": conc, "film": film, "pO2": pO2,
                             "obs_mask": mask})
    return pool


def _forward_prop_vec(cond, theta, db):
    c = _call(theta, cond)
    r = ff.forward(cond["dopant"], cond.get("conc", 0.0), T_anneal=c["T_anneal"], pO2=c["pO2"],
                   film=cond.get("film", True), E_B_eV=c["E_B"], Nd_bg=c["Nd_bg"], db=db)
    v = []
    for k in PROP_KEYS:
        val = r.get(k)
        v.append(np.log10(max(val, 1e-30)) if k in PROP_LOG else (float(val) if val is not None else 0.0))
    return np.array(v)


def property_jacobian(cond, theta, db=None):
    """d(9 properties)/d(theta) at a condition, scored space (log for multiplicative). (9 x |theta|)."""
    db = db or kroger_db.load()
    J = np.zeros((len(PROP_KEYS), len(THETA_KEYS)))
    for j, key in enumerate(THETA_KEYS):
        tp = dict(theta); tm = dict(theta)
        tp[key] += FD_STEP[key]; tm[key] -= FD_STEP[key]
        J[:, j] = (_forward_prop_vec(cond, tp, db) - _forward_prop_vec(cond, tm, db)) / (2 * FD_STEP[key])
    return J


def predictive_Phi(targets, cov, theta, weights=None, db=None, target_jacs=None):
    """Prediction-weighted A/L-optimal objective: Σ_targets Σ_props w_j · J_j Σ J_jᵀ.
    Lower = tighter predictions at the lab's TARGET conditions. Capped props (w=0) are ignored."""
    db = db or kroger_db.load()
    w = weights or DESIGN_WEIGHTS
    total = 0.0
    for i, x in enumerate(targets):
        Jt = target_jacs[i] if target_jacs is not None else property_jacobian(x, theta, db)
        for j, k in enumerate(PROP_KEYS):
            if w.get(k, 0.0) > 0:
                total += w[k] * float(Jt[j] @ cov @ Jt[j])
    return total


def implausibility(points, theta, db=None, discrepancy=None):
    """NROY / history-matching gate (agent-#3 B2 robustness). For the best-fit theta, the max
    standardized residual over the lab's points: I = |meas - sim| / sqrt(noise^2 + discrepancy^2).
    I>3 for ALL plausible theta => no theta reconciles sim with this bench => report relative-only."""
    db = db or kroger_db.load()
    disc = discrepancy or {"log10_n": 0.5, "log10_mu": 0.3, "log10_sigma": 0.5}
    worst = 0.0
    for pt in points:
        obs = pt.get("obs_mask") or [o for o in OBS_KEYS if o in pt["meas"]]
        idx = [OBS_KEYS.index(o) for o in obs]
        pred = observe(pt, theta, db)[idx]
        for m, o in enumerate(obs):
            sd = np.sqrt(OBS_NOISE[o] ** 2 + disc.get(o, 0.5) ** 2)
            worst = max(worst, abs(pred[m] - pt["meas"][o]) / sd)
    return {"max_implausibility": round(float(worst), 2),
            "NROY_nonempty": bool(worst <= 3.0),
            "verdict": ("theta reconciles the bench (calibratable absolutes OK)" if worst <= 3.0
                        else "NO theta reconciles sim+bench -> report RELATIVE-ONLY, do not emit "
                             "calibrated absolutes (consistent with the ceiling finding)")}


def predict_with_uncertainty(cond, theta, cov, db=None):
    """Propagate the theta posterior to the 9 properties via symmetric sigma-points (no RNG).
    Returns per-property median + 1-sigma band — the calibrated, genuinely-simulated bar."""
    db = db or kroger_db.load()
    mean = np.array([theta[k] for k in THETA_KEYS])
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(THETA_KEYS)))
    keys = ["hall_n_cm3", "hall_mu_cm2Vs", "sigma_S_cm", "Eg_optical_eV", "dEg_eV",
            "V_O_cm3_quench", "dark_activation_eV", "PDR_score", "tau_score"]
    pts = [mean]
    for j in range(len(THETA_KEYS)):
        e = np.zeros(len(THETA_KEYS)); e[j] = 1.0
        pts += [mean + L @ e, mean - L @ e]
    samples = {k: [] for k in keys}
    for p in pts:
        th = {k: float(p[i]) for i, k in enumerate(THETA_KEYS)}
        c = _call(th, cond)
        r = ff.forward(cond["dopant"], cond.get("conc", 0.0), T_anneal=c["T_anneal"], pO2=c["pO2"],
                       film=cond.get("film", True), E_B_eV=c["E_B"], Nd_bg=c["Nd_bg"], db=db)
        for k in keys:
            samples[k].append(r.get(k))
    out = {}
    for k in keys:
        arr = np.array([s for s in samples[k] if s is not None], dtype=float)
        if k in ("hall_n_cm3", "sigma_S_cm", "V_O_cm3_quench"):
            la = np.log10(np.clip(arr, 1e-30, None))
            out[k] = {"median": float(10 ** np.median(la)), "band_dex": round(float(la.std()), 2)}
        else:
            out[k] = {"median": round(float(np.median(arr)), 4), "band": round(float(arr.std()), 4)}
    return out
