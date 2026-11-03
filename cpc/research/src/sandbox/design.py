"""Exploratory local experimental design with explicit observation assumptions.

The prior, noise scales, parameter bounds and model are assumptions, not validated
coverage statements. Calibration does not prove parameter identification or an actual
growth history. The default observables are log Hall density and log Hall mobility.
Derived conductivity is not counted as a third independent measurement. Using all three
requires a declared independent conductivity measurement and its full noise covariance.
Historical performance and universal prediction-ceiling claims were withdrawn in the
2026-10-15 review revision.
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
DEFAULT_OBS = ["log10_n", "log10_mu"]
OBS_NOISE = {"log10_n": 0.15, "log10_mu": 0.10, "log10_sigma": 0.20}   # assumed dex scales

# Illustrative objective weights; they are not measured utility or validated capability grades.
PROP_KEYS = ["hall_density_cm3", "hall_mu_cm2Vs", "sigma_S_cm", "Eg_optical_eV", "dEg_eV",
             "V_O_cm3_quench", "dark_activation_eV", "PDR_score", "tau_score"]
PROP_LOG = {"hall_density_cm3", "sigma_S_cm", "V_O_cm3_quench"}   # scored in log10 (multiplicative)
DESIGN_WEIGHTS = {"hall_density_cm3": 1.0, "sigma_S_cm": 0.7, "hall_mu_cm2Vs": 0.3,
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
    return np.array([np.log10(max(r["hall_density_cm3"], 1.0)),
                     np.log10(max(r["hall_mu_cm2Vs"], 1e-3)),
                     np.log10(max(r["sigma_S_cm"], 1e-30))])


def jacobian(cond, theta, obs_mask=None, db=None):
    db = db or kroger_db.load()
    idx = [OBS_KEYS.index(o) for o in (obs_mask or DEFAULT_OBS)]
    J = np.zeros((len(idx), len(THETA_KEYS)))
    for j, key in enumerate(THETA_KEYS):
        tp = dict(theta); tm = dict(theta)
        tp[key] += FD_STEP[key]; tm[key] -= FD_STEP[key]
        J[:, j] = (observe(cond, tp, db)[idx] - observe(cond, tm, db)[idx]) / (2 * FD_STEP[key])
    return J


def _observation_contract(cond, obs_mask=None):
    """Return observables and a positive-definite covariance in their listed order."""
    explicit = obs_mask if obs_mask is not None else cond.get('obs_mask')
    if explicit is None:
        obs = [o for o in OBS_KEYS if o in cond['meas']] if 'meas' in cond else list(DEFAULT_OBS)
        if set(OBS_KEYS).issubset(obs) and not cond.get('conductivity_is_independent', False):
            obs.remove('log10_sigma')
    else:
        obs = list(explicit)
    if not obs or len(set(obs)) != len(obs) or any(o not in OBS_KEYS for o in obs):
        raise ValueError('Supply distinct known observation keys')
    if set(OBS_KEYS).issubset(obs):
        if not cond.get('conductivity_is_independent', False) or 'obs_cov' not in cond:
            raise ValueError('All three observables require independently measured conductivity and obs_cov; derived conductivity cannot add information')
    cov = np.asarray(cond.get('obs_cov', np.diag([OBS_NOISE[o]**2 for o in obs])), dtype=float)
    if cov.shape != (len(obs),len(obs)) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T):
        raise ValueError('obs_cov must be a finite symmetric covariance in obs_mask order')
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as ex:
        raise ValueError('obs_cov must be positive definite; remove algebraically derived observations') from ex
    return obs, cov


def fisher(cond, theta, obs_mask=None, db=None):
    obs, cov = _observation_contract(cond, obs_mask)
    J = jacobian(cond, theta, obs, db)
    return J.T @ np.linalg.solve(cov,J)


def prior_precision():
    return np.diag([1.0 / THETA_PRIOR_STD[k] ** 2 for k in THETA_KEYS])


def theta_fit(points, theta_init=None, db=None, iters=12):
    """Fit per-lab theta by Gauss-Newton on the lab's own measurements + prior.
    points: [{dopant, conc, film, pO2?, obs_mask?, meas:{log10_n, log10_mu?, log10_sigma?}}].
    Returns (theta_hat, cov). Covariance is a local prior-regularized approximation;
    invertibility does not imply identification by the observations."""
    db = db or kroger_db.load()
    theta = dict(theta_init or THETA0)
    P0 = prior_precision()
    m0 = np.array([THETA0[k] for k in THETA_KEYS])
    JtWJ = P0.copy()
    for _ in range(iters):
        JtWr = np.zeros(len(THETA_KEYS)); JtWJ = P0.copy()
        for pt in points:
            obs, obs_cov = _observation_contract(pt)
            idx = [OBS_KEYS.index(o) for o in obs]
            pred = observe(pt, theta, db)[idx]
            meas = np.array([pt["meas"][o] for o in obs])
            J = jacobian(pt, theta, obs, db)
            Winv = np.linalg.inv(obs_cov)
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
    # Recompute at the final state, not the state before the last update.
    return theta, np.linalg.inv(_info_matrix(points, theta, db))


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
    representative = {"dopant":"Si","conc":0.005,"film":True,"pO2":1e-5}
    jn = jacobian(representative,theta,["log10_n"],db)[0]
    cov_before = np.linalg.inv(_info_matrix(base_points,theta,db))
    cov_after = np.linalg.inv(M)
    return {
        "criterion": criterion,
        "battery": [pool[i] for i in chosen],
        "info_gain": gains,          # per-step improvement in the chosen criterion's score
        "theta_std_before": {k: round(float(std_before[j]), 3) for j, k in enumerate(THETA_KEYS)},
        "theta_std_after": {k: round(float(std_after[j]), 3) for j, k in enumerate(THETA_KEYS)},
        "abs_n_dex_before": float(np.sqrt(max(jn@cov_before@jn,0.))),
        "abs_n_dex_after": float(np.sqrt(max(jn@cov_after@jn,0.))),
        "validation_status":"unvalidated local design calculation",
        "note":"abs_n_dex fields are linearized parameter-only standard deviations at the stated "
               "representative condition, including parameter covariance. They exclude discrepancy "
               "and measurement prediction noise and have no demonstrated interval coverage.",
        "representative_condition":representative,
    }


def _dlogn_dlogTf(theta, db, cond=None):
    cond = cond or {"dopant": "Si", "conc": 0.005, "film": True, "pO2": 1e-5}
    J = jacobian(cond, theta, ["log10_n"], db)
    return abs(float(J[0, THETA_KEYS.index("log10_Tf")]))


def default_pool():
    """Illustrative candidate conditions; practical feasibility and identifiability need checking."""
    pool = []
    for dop, conc, mask in [
        ("undoped", 0.0, ["log10_n"]),
        ("Si", 0.001, DEFAULT_OBS), ("Si", 0.005, DEFAULT_OBS), ("Si", 0.02, DEFAULT_OBS),
        ("Sn", 0.005, DEFAULT_OBS), ("Hf", 0.005, DEFAULT_OBS)]:
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
    """Marginal standardized residual at one supplied theta, not a search over parameter space.
    The tolerance and discrepancy are assumed. Failure at this point says nothing about whether
    another parameter setting can explain the observations."""
    db = db or kroger_db.load()
    disc = discrepancy or {"log10_n": 0.5, "log10_mu": 0.3, "log10_sigma": 0.5}
    worst = 0.0
    for pt in points:
        obs, obs_cov = _observation_contract(pt)
        idx = [OBS_KEYS.index(o) for o in obs]
        pred = observe(pt, theta, db)[idx]
        for m, o in enumerate(obs):
            sd = np.sqrt(obs_cov[m,m] + disc.get(o, 0.5) ** 2)
            worst = max(worst, abs(pred[m] - pt["meas"][o]) / sd)
    return {"max_implausibility": round(float(worst), 2),
            "tested_theta_within_assumed_tolerance":bool(worst <= 3.),
            "NROY_nonempty": True if worst <= 3. else None,
            "verdict": ("Supplied theta meets the assumed marginal tolerance; prediction accuracy is unvalidated"
                        if worst <= 3. else "Supplied theta fails the assumed tolerance; no parameter-space exclusion was performed")}


def predict_with_uncertainty(cond, theta, cov, db=None):
    """Linearized parameter propagation J cov J^T; no predictive coverage is claimed.

    Historical band fields are None. The former standard deviation of equally weighted
    central/axis points was not a correctly weighted Gaussian moment or calibrated interval.
    """
    db = db or kroger_db.load()
    cov = np.asarray(cov,dtype=float)
    if cov.shape != (len(THETA_KEYS),len(THETA_KEYS)) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T):
        raise ValueError('Parameter covariance must be finite and symmetric with one row per THETA_KEYS')
    if np.linalg.eigvalsh(cov).min() < -1e-12:
        raise ValueError('Parameter covariance must be positive semidefinite')
    center = _forward_prop_vec(cond,theta,db)
    jac = property_jacobian(cond,theta,db)
    out = {}
    for i,k in enumerate(PROP_KEYS):
        std=float(np.sqrt(max(jac[i]@cov@jac[i],0.)))
        out[k]={"value":float(10**center[i] if k in PROP_LOG else center[i]),
                "linearized_parameter_std":std,"std_units":"dex" if k in PROP_LOG else "property units",
                "band":None,"band_dex":None,"range":None,
                "validation_status":"unvalidated; parameter propagation only; excludes model discrepancy and predictive measurement noise"}
    return out
