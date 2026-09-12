"""R8 follow-on to ICC-05 + PCV-29, on the CURRENT corpus (published values predate the
Phase-105 lab-row correction). Replicates the phase107 protocol exactly:
  (a) conditional ICC = intercept-only REML ICC of the OUT-OF-FOLD ridge residual
      (study-grouped 5-fold CV, 5 seeds -- leakage-free, so covariates cannot proxy the lab);
  (b) dopant-fixed-effects ICC via mixedlm y ~ dopant + (1|study);
  (c) PCV-29: same OOF conditioning for dark/PDR with the measurement-protocol covariates
      (bias_V, wavelength_nm from the loaded FOM table) added to the process covariates.
Writes results/tier2/r8_icc_conditional.json.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("c72", NET / "scripts/_phase72_common.py")
c72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c72)
helpers = {"__file__": str(PROJ / "scripts/r8_icc_lrt.py")}
exec((PROJ / "scripts/r8_icc_lrt.py").read_text().split("NSIM = 2000")[0], helpers)
suffstats, reml_fit = helpers["suffstats"], helpers["reml_fit"]

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

FOMS = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
        "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]


def icc_of(y, g):
    return reml_fit(suffstats(np.asarray(y, float), np.asarray(g)))["icc"]


def oof_residualize(y, X, groups, n_seeds=5):
    n = len(y)
    acc = np.zeros(n)
    for s in range(n_seeds):
        fo = c72.folds(groups, s)
        pred = np.full(n, np.nan)
        for f in np.unique(fo):
            tr, te = fo != f, fo == f
            if tr.sum() < 5 or te.sum() == 0:
                pred[te] = y[tr].mean() if tr.sum() else y.mean()
                continue
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        acc += pred
    return y - acc / n_seeds


import statsmodels.formula.api as smf

out = {}
deltas = []
for fom in FOMS:
    d = c72.load_fom(fom)
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"]).astype(str)
    F = np.asarray(d["F"], float)
    ok = np.isfinite(y) & np.all(np.isfinite(F), axis=1)
    yy, gg, FF = y[ok], g[ok], F[ok]
    keep = FF.std(axis=0) > 1e-9
    FF = FF[:, keep]
    icc0 = icc_of(yy, gg)
    icc_c = icc_of(oof_residualize(yy, FF, gg), gg)
    # dopant fixed effects via mixedlm
    sub = d["sub"].reset_index(drop=True)
    col = next((c for c in sub.columns if "element" in c.lower() or "dopant" in c.lower()), None)
    el = sub[col].astype(str).to_numpy()[np.isfinite(y)] if col else None
    icc_fe = None
    if el is not None:
        df = pd.DataFrame({"y": y[np.isfinite(y)], "study": g[np.isfinite(y)], "el": el})
        try:
            m = smf.mixedlm("y ~ C(el)", df, groups=df["study"]).fit(reml=True)
            icc_fe = float(m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale))
        except Exception:
            pass
    out[fom] = {"icc": round(icc0, 4), "icc_conditional_oof": round(icc_c, 4),
                "icc_dopant_fe_mixedlm": None if icc_fe is None else round(icc_fe, 4),
                "delta_conditional_pp": round(100 * (icc0 - icc_c), 1)}
    deltas.append(icc0 - icc_c)
    print(fom, out[fom], flush=True)

out["_median_conditional_removal_pp"] = round(100 * float(np.median(deltas)), 1)
fes = [v["icc_dopant_fe_mixedlm"] for v in out.values()
       if isinstance(v, dict) and v.get("icc_dopant_fe_mixedlm") is not None]
out["_dopant_fe_range"] = [round(min(fes), 3), round(max(fes), 3)]

# ---- PCV-29: protocol covariates (from the loaded FOM tables) ----
pcv = {}
for fom in ("dark_current_pA", "photo_dark_ratio"):
    d = c72.load_fom(fom)
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"]).astype(str)
    F = np.asarray(d["F"], float)
    sub = d["sub"].reset_index(drop=True)
    proto = [c for c in ("bias_V", "wavelength_nm") if c in sub.columns]
    P = sub[proto].astype(float).to_numpy() if proto else np.zeros((len(y), 0))
    coverage = {c: round(float(np.isfinite(P[:, i]).mean()), 3) for i, c in enumerate(proto)}
    ok = np.isfinite(y) & np.all(np.isfinite(F), axis=1) & np.all(np.isfinite(P), axis=1)
    if ok.sum() >= 30:
        Xall = np.column_stack([F[ok][:, F[ok].std(axis=0) > 1e-9], P[ok]])
        icc_cov_rows = icc_of(y[ok], g[ok])
        icc_p = icc_of(oof_residualize(y[ok], Xall, g[ok]), g[ok])
    else:
        icc_cov_rows, icc_p = None, None
    pcv[fom] = {"protocol_covariates": proto, "coverage": coverage, "n_covered": int(ok.sum()),
                "icc_covered_rows": None if icc_cov_rows is None else round(icc_cov_rows, 4),
                "icc_after_process_plus_protocol_oof": None if icc_p is None else round(icc_p, 4)}
    print(fom, pcv[fom], flush=True)
out["pcv29_protocol_conditioning"] = pcv

(PROJ / "results/tier2/r8_icc_conditional.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_icc_conditional.json")
