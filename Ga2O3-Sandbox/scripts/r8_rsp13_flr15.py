"""R8 RSP-13 + FLR-15 (major).
RSP-13 (mobility): (i) paired Wilcoxon over per-lab LOLO medians, hybrid vs corpus-median
constant; (ii) LOLO response metric: within-held-out-lab Spearman(pred, meas) over labs with
>=3 distinct-mu rows (a constant predictor has no within-lab response by construction);
(iii) film/crystal stratified LOLO medians.
FLR-15: the designer's within-lab floor recomputed leave-one-point-out over the 13 dedup labs
(the published 0.47 is the fit-on-all-points floor).
Writes results/tier2/r8_rsp13_flr15.json."""
import sys, json, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(PROJ / "src"))


def _imp(n, f):
    s = importlib.util.spec_from_file_location(n, f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


out = {}
# ---------- RSP-13 ----------
from sandbox import full_forward, kroger_db
db = kroger_db.load()
hc = _imp("hc", PROJ / "scripts/hybrid_certify.py")
tp = pd.read_csv(PROJ.parent / "Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
DOP = set(db.elements) - {"Ga", "O"}
rows = []
for _, r in tp.iterrows():
    el = str(r["dopant_element"])
    if el not in DOP or pd.isna(r.get("mu_cm2Vs")):
        continue
    cf = hc.conc_frac(r.get("concentration_value"), r.get("concentration_unit"))
    if cf is None or cf <= 0:
        cf = 0.005
    film = "sputter" in str(r.get("growth_method", "")).lower()
    o = full_forward.forward(el, cf, T_anneal=hc.Tf if hasattr(hc, "Tf") else 1500.0,
                             pO2=1e-5, film=film, db=db)
    pred, meas = o["hall_mu_cm2Vs"], float(r["mu_cm2Vs"])
    if pred > 0 and meas > 0:
        rows.append({"doi": str(r.get("doi", "")), "el": el, "film": film,
                     "lp": np.log10(pred), "lm": np.log10(meas)})
df = pd.DataFrame(rows)
df["resid"] = df.lp - df.lm
labs = df.doi.unique()
ph, pc, resp, strata = [], [], [], {"film": [], "crystal": []}
for held in labs:
    tr, te = df[df.doi != held], df[df.doi == held]
    off = tr.resid.median()
    e_h = np.abs(te.resid - off)
    const = tr.lm.median()                      # corpus-median constant from other labs
    e_c = np.abs(te.lm - const)
    ph.append(float(e_h.median())); pc.append(float(e_c.median()))
    strata["film" if te.film.mode()[0] else "crystal"].append(float(e_h.median()))
    if len(te) >= 3 and te.lm.nunique() >= 3:
        rho = stats.spearmanr(te.lp, te.lm).statistic
        if np.isfinite(rho):
            resp.append(float(rho))
w = stats.wilcoxon(ph, pc, alternative="less")
out["rsp13"] = {
    "n_labs": len(labs),
    "hybrid_median_of_perlab_medians_dex": round(float(np.median(ph)), 3),
    "constant_median_of_perlab_medians_dex": round(float(np.median(pc)), 3),
    "wilcoxon_p_one_sided": round(float(w.pvalue), 4),
    "n_labs_hybrid_better": int(np.sum(np.array(ph) < np.array(pc))),
    "response_labs_n": len(resp),
    "within_heldout_lab_spearman_median": round(float(np.median(resp)), 3) if resp else None,
    "within_heldout_lab_spearman_frac_positive": round(float(np.mean(np.array(resp) > 0)), 3) if resp else None,
    "strata_median_dex": {k: round(float(np.median(v)), 3) if v else None
                          for k, v in strata.items()},
    "strata_n_labs": {k: len(v) for k, v in strata.items()},
    "constant_response": 0.0}
print("RSP-13:", json.dumps(out["rsp13"]), flush=True)

# ---------- FLR-15 ----------
dv = _imp("dv", PROJ / "scripts/design_validate.py")
DUPS = {'10.48550/arxiv.2311.00821', '10.48550/arxiv.2407.17089', '10.48550/arxiv.2001.11187'}
labs2 = {k: v for k, v in dv.load_labs().items() if k != 'nan' and not any(k.startswith(x) for x in DUPS)}
floor_in, floor_loo = [], []
for doi, pts in labs2.items():
    doped = [i for i, p in enumerate(pts) if not p["und"]]
    if len(doped) < 4:
        continue
    use_nd = any(p["und"] for p in pts)
    floor_in.append(dv.score(dv.fit_grid(pts, use_nd), [pts[i] for i in doped]))
    errs = []
    for i in doped:
        rest = [pts[j] for j in range(len(pts)) if j != i]
        errs.append(dv.score(dv.fit_grid(rest, use_nd), [pts[i]]))
    floor_loo.append(float(np.median(errs)))
out["flr15"] = {"floor_insample_pooled_median": round(float(np.median(floor_in)), 3),
                "floor_leave_one_point_out_pooled_median": round(float(np.median(floor_loo)), 3),
                "n_labs": len(floor_loo)}
print("FLR-15:", json.dumps(out["flr15"]), flush=True)

(PROJ / "results/tier2/r8_rsp13_flr15.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_rsp13_flr15.json")
