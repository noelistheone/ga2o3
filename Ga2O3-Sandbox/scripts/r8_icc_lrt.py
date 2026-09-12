"""R8 ICC-05 (blocking): put the 58-85% ceiling on exact-test footing.
Per property (8 direct + derived conductivity):
  (1) exact restricted LRT of sigma_b^2 > 0 (parametric-bootstrap null a la Crainiceanu-Ruppert,
      2000 sims: fit H0 (iid), simulate, refit both, P(LRT* >= LRT_obs));
  (2) parametric-bootstrap ICC percentile CI (2000 resamples from the FITTED mixed model,
      preserving the observed group-size structure);
  (3) cluster-size distribution (n_studies, median size, singleton fraction);
  (4) leave-one-study-out ICC influence (max |delta ICC|).
Custom closed-form-profiled REML (intercept-only) so 2000x9 refits run in seconds.
Writes results/tier2/r8_icc_lrt.json.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("c72", NET / "scripts/_phase72_common.py")
c72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c72)

FOMS = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
        "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]


def suffstats(y, g):
    stats = []
    for grp in np.unique(g):
        v = y[g == grp]
        stats.append((len(v), float(v.mean()), float(((v - v.mean()) ** 2).sum())))
    return stats


def reml_fit(stats):
    """Profile REML over rho = sigma_b^2/sigma_e^2 (1-D), closed-form sigma_e^2 and mu."""
    N = sum(n for n, _, _ in stats)
    q = len(stats)

    def neg_reml(log_rho):
        rho = np.exp(log_rho)
        # weights for mu-hat: w_j = n_j / (1 + rho n_j)  (in units of 1/sigma_e^2)
        w = np.array([n / (1 + rho * n) for n, _, _ in stats])
        yb = np.array([m for _, m, _ in stats])
        mu = (w * yb).sum() / w.sum()
        # quadratic form / sigma_e^2
        Q = sum(ss for _, _, ss in stats) + (w * (yb - mu) ** 2).sum()
        s2e = Q / (N - 1)
        ll = -0.5 * (sum((n - 1) * np.log(s2e) + np.log(s2e * (1 + rho * n)) for n, _, _ in stats)
                     + Q / s2e + np.log((w / s2e).sum()))
        return -ll

    r = minimize_scalar(neg_reml, bounds=(-14, 8), method="bounded",
                        options={"xatol": 1e-7})
    rho = float(np.exp(r.x))
    # recover components at the optimum
    w = np.array([n / (1 + rho * n) for n, _, _ in stats])
    yb = np.array([m for _, m, _ in stats])
    mu = (w * yb).sum() / w.sum()
    Q = sum(ss for _, _, ss in stats) + (w * (yb - mu) ** 2).sum()
    s2e = Q / (N - 1)
    s2b = rho * s2e
    # boundary handling: rho at lower bound => sigma_b ~ 0
    if rho < 2e-6:
        s2b = 0.0
    icc = s2b / (s2b + s2e) if (s2b + s2e) > 0 else 0.0
    return {"mu": float(mu), "s2b": float(s2b), "s2e": float(s2e), "icc": float(icc),
            "neg_reml": float(r.fun)}


def reml_ll_null(stats):
    """REML log-lik under H0 (sigma_b^2 = 0): iid normal, REML for (mu, sigma^2)."""
    N = sum(n for n, _, _ in stats)
    yb = np.array([m for _, m, _ in stats])
    n = np.array([x for x, _, _ in stats])
    mu = (n * yb).sum() / N
    Q = sum(ss for _, _, ss in stats) + (n * (yb - mu) ** 2).sum()
    s2 = Q / (N - 1)
    ll = -0.5 * (N * np.log(s2) + Q / s2 + np.log(N / s2))
    return float(ll)


def simulate(stats, mu, s2b, s2e, rng):
    out = []
    for n, _, _ in stats:
        b = rng.normal(0, np.sqrt(s2b)) if s2b > 0 else 0.0
        v = mu + b + rng.normal(0, np.sqrt(s2e), n)
        out.append((n, float(v.mean()), float(((v - v.mean()) ** 2).sum())))
    return out


NSIM = 2000
rng = np.random.default_rng(2026)
res = {}


def analyze(name, y, g):
    y = np.asarray(y, float)
    g = np.asarray(g)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    stats = suffstats(y, g)
    fit = reml_fit(stats)
    lrt_obs = 2 * (-fit["neg_reml"] - reml_ll_null(stats))
    # (1) exact restricted LRT null via parametric bootstrap under H0
    null = np.empty(NSIM)
    for i in range(NSIM):
        st = simulate(stats, fit["mu"], 0.0, fit["s2b"] + fit["s2e"], rng)
        f1 = reml_fit(st)
        null[i] = 2 * (-f1["neg_reml"] - reml_ll_null(st))
    p_lrt = float((np.sum(null >= max(lrt_obs, 0)) + 1) / (NSIM + 1))
    # (2) parametric-bootstrap ICC CI from the fitted model
    boot = np.empty(NSIM)
    for i in range(NSIM):
        st = simulate(stats, fit["mu"], fit["s2b"], fit["s2e"], rng)
        boot[i] = reml_fit(st)["icc"]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    # (3) cluster sizes
    sizes = np.array([n for n, _, _ in stats])
    # (4) leave-one-study-out influence
    iccs = []
    for j in range(len(stats)):
        iccs.append(reml_fit(stats[:j] + stats[j + 1:])["icc"])
    dmax = float(np.max(np.abs(np.array(iccs) - fit["icc"])))
    r = {"n_rows": int(len(y)), "n_studies": int(len(stats)),
         "icc_reml": round(fit["icc"], 4), "lrt_stat": round(float(max(lrt_obs, 0)), 3),
         "p_exact_restricted_lrt": p_lrt, "icc_ci95_parametric_bootstrap": [round(c, 3) for c in ci],
         "singleton_fraction": round(float((sizes == 1).mean()), 3),
         "median_cluster_size": float(np.median(sizes)),
         "loo_study_max_abs_dicc": round(dmax, 4),
         "mc_se_at_p": round(float(np.sqrt(p_lrt * (1 - p_lrt) / NSIM)), 5)}
    print(name, r, flush=True)
    return r


spec109 = importlib.util.spec_from_file_location("p109", NET / "scripts/phase109_conductivity.py")
p109 = importlib.util.module_from_spec(spec109)
spec109.loader.exec_module(p109)

for fom in FOMS:
    d = c72.load_fom(fom)
    res[fom] = analyze(fom, np.asarray(d["y"], float), np.asarray(d["groups"]).astype(str))

dsig = p109.load_conductivity()
res["conductivity_sigma"] = analyze("conductivity_sigma", np.asarray(dsig["y"], float),
                                    np.asarray(dsig["groups"]).astype(str))

res["_method"] = ("intercept-only REML (1-D profiled), exact restricted LRT of sigma_b^2>0 via "
                  "2000-sim parametric bootstrap under H0 (Crainiceanu-Ruppert-style), ICC CI via "
                  "2000-sim parametric bootstrap from the fitted model, seed 2026")
(PROJ / "results/tier2/r8_icc_lrt.json").write_text(json.dumps(res, indent=1))
print("wrote results/tier2/r8_icc_lrt.json")
