"""R8: dopant-fixed-effects ICC for all 8 properties via custom REML with fixed effects
(y ~ dopant + (1|study)); mixedlm failed to converge on half the tables. Profiled 1-D REML
over rho with exact GLS for the dopant-dummy fixed effects. Writes results/tier2/r8_icc_dopantfe.json."""
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


def reml_fe(y, g, X):
    """REML for y = X b + u_g + e, profiled over rho = s2b/s2e.
    Per-group Woodbury: V_j^{-1} = (I - rho/(1+rho n_j) J)/s2e."""
    groups = np.unique(g)
    parts = [(np.asarray(y[g == gr], float), np.asarray(X[g == gr], float)) for gr in groups]
    N, p = len(y), X.shape[1]

    def crit(log_rho):
        rho = np.exp(log_rho)
        XtX = np.zeros((p, p)); Xty = np.zeros(p); yty = 0.0; logdetV = 0.0
        for yj, Xj in parts:
            nj = len(yj)
            c = rho / (1 + rho * nj)
            sx = Xj.sum(0); sy = yj.sum()
            XtX += Xj.T @ Xj - c * np.outer(sx, sx)
            Xty += Xj.T @ yj - c * sx * sy
            yty += yj @ yj - c * sy * sy
            logdetV += np.log1p(rho * nj)
        b = np.linalg.solve(XtX + 1e-10 * np.eye(p), Xty)
        Q = yty - b @ Xty
        s2e = Q / (N - p)
        sgn, logdetX = np.linalg.slogdet(XtX / s2e)
        return -(-0.5 * ((N - p) * np.log(s2e) + logdetV + Q / s2e + logdetX)), (b, s2e, Q)

    r = minimize_scalar(lambda lr: crit(lr)[0], bounds=(-14, 8), method="bounded",
                        options={"xatol": 1e-7})
    rho = float(np.exp(r.x))
    _, (b, s2e, Q) = crit(r.x)
    s2b = 0.0 if rho < 2e-6 else rho * s2e
    return s2b / (s2b + s2e)


out = {}
for fom in FOMS:
    d = c72.load_fom(fom)
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"]).astype(str)
    el = np.asarray(d["labels"]).astype(str)
    ok = np.isfinite(y)
    y, g, el = y[ok], g[ok], el[ok]
    # dopant one-hots (incl. intercept via full dummies of all levels, drop-first + intercept)
    levels = sorted(set(el))
    X = np.column_stack([np.ones(len(y))] +
                        [(el == lv).astype(float) for lv in levels[1:]])
    icc0_helpers = {"__file__": str(PROJ / "scripts/r8_icc_lrt.py")}
    exec((PROJ / "scripts/r8_icc_lrt.py").read_text().split("NSIM = 2000")[0], icc0_helpers)
    icc0 = icc0_helpers["reml_fit"](icc0_helpers["suffstats"](y, g))["icc"]
    icc_fe = reml_fe(y, g, X)
    out[fom] = {"icc": round(icc0, 4), "icc_dopant_fe": round(float(icc_fe), 4),
                "delta": round(float(icc_fe - icc0), 4), "n_dopant_levels": len(levels)}
    print(fom, out[fom], flush=True)

vals = [v["icc_dopant_fe"] for v in out.values()]
out["_range"] = [round(min(vals), 3), round(max(vals), 3)]
(PROJ / "results/tier2/r8_icc_dopantfe.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_icc_dopantfe.json")
