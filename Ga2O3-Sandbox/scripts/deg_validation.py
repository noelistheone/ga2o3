"""Validate the calibrated optical layer's dEg (Burstein-Moss widening) against the corpus dEg —
the parent project's STRONGEST ML property (within-paper ρ 0.617). Within-paper doped-vs-undoped
Eg contrasts only (lab confound cancels). Tests sign + Spearman of predicted vs observed dEg.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db  # noqa: E402

db = kroger_db.load()
KROGER = set(db.elements)
PROC = dict(T_anneal=1350.0, pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)

eg = pd.read_csv(NET / "results/phase69/eg_extracted.csv")
print("Eg corpus cols:", list(eg.columns)[:16])
# identify element + Eg + doi columns
col_el = "dopant_element"
col_eg = "eg_eV"
col_doi = "file"        # group by source paper
col_conc = "concentration_value"
print(f"using element={col_el} eg={col_eg} group={col_doi} host_flag=is_host_reference")

und = ff.forward("undoped", 0.0, db=db, **PROC)
cache = {"n_und": und["hall_n_cm3"]}
pred_deg = {}
for X in [e for e in db.elements if e not in ("Ga", "O")]:
    try:
        r = ff.forward(X, 0.01, db=db, undoped_cache=cache, **PROC)
        pred_deg[X] = r["dEg_eV"]
    except Exception:
        pass

pairs = []
if col_el and col_eg and col_doi:
    for doi, g in eg.groupby(col_doi):
        if "is_host_reference" in g.columns:
            und_rows = g[g["is_host_reference"].fillna(False).astype(bool)]
        else:
            und_rows = g[g[col_el].astype(str).str.lower().isin(["undoped", "none", "pristine", "intrinsic"])]
        if und_rows.empty or und_rows[col_eg].isna().all():
            continue
        eg_und = float(und_rows[col_eg].astype(float).median())
        for _, row in g.iterrows():
            el = str(row[col_el])
            if (row.get("is_host_reference", False) or el.lower() in
                    ("undoped", "none", "pristine", "intrinsic", "nan") or el not in pred_deg):
                continue
            if pd.isna(row[col_eg]):
                continue
            obs = float(row[col_eg]) - eg_und
            if abs(obs) < 0.01:
                continue
            pairs.append({"doi": str(doi)[:28], "el": el, "obs_dEg": round(obs, 3),
                          "pred_dEg": round(pred_deg[el], 4)})

report = {"n_pairs": len(pairs)}
if pairs:
    obs = np.array([p["obs_dEg"] for p in pairs])
    prd = np.array([p["pred_dEg"] for p in pairs])
    sign_match = int(np.sum(np.sign(obs) == np.sign(prd)))
    frac_up_obs = float(np.mean(obs > 0))
    rho = float(spearmanr(prd, obs).correlation) if len(pairs) >= 4 else None
    report.update({
        "sign_match": f"{sign_match}/{len(pairs)}",
        "frac_correct": round(sign_match / len(pairs), 3),
        "corpus_frac_up": round(frac_up_obs, 3),
        "majority_baseline": round(max(frac_up_obs, 1 - frac_up_obs), 3),
        "beats_majority": bool(sign_match / len(pairs) > max(frac_up_obs, 1 - frac_up_obs)),
        "spearman_pred_vs_obs": round(rho, 3) if rho is not None else None,
        "note": "predicted dEg is BM widening for degenerate donors (~+0.006..+0.028 eV); most "
                "corpus dEg contrasts also positive → check beats_majority for real skill",
        "examples": pairs[:15],
    })
(PROJ / "results/tier2/deg_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
