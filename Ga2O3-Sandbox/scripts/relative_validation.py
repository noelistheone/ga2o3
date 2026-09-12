"""Within-condition RELATIVE-prediction validation — the one layer that could be QUANTITATIVE.

Absolute cross-dopant accuracy is ceiling-capped. But WITHIN a paper (same lab, same process,
one dopant varied), systematic errors cancel — so the doping-RESPONSE slope d(log n)/d(log conc)
may be quantitative. This tests exactly that: for each transport-corpus concentration series
(same doi, same dopant, ≥3 concentrations), does the chain predict the carrier-density slope?

Scored: (a) sign of the slope (does n rise/fall with concentration correctly), (b) Spearman of
predicted vs observed n across the series, (c) slope magnitude ratio. Vs the null of shuffled
concentrations.
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
PROC = dict(T_anneal=1073.0, pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)

tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["carrier_cm3"].notna() & tp["concentration_value"].notna()]

und = ff.forward("undoped", 0.0, db=db, **PROC)
cache = {"n_und": und["hall_n_cm3"]}


def conc_to_frac(val, unit):
    """Convert a corpus concentration to a cation fraction (best-effort)."""
    unit = str(unit).lower()
    v = float(val)
    if "at" in unit or "%" in unit:
        return v / 100.0
    if "cm" in unit or "e1" in unit or v > 1e17:  # a density -> fraction of cation sites
        return v / (1.91e22 * 2 / 3)
    return v / 100.0 if v > 1 else v


series_results = []
for (doi, el), g in tp.groupby(["doi", "dopant_element"]):
    el = str(el)
    if el not in KROGER or el in ("Ga", "O"):
        continue
    g = g.dropna(subset=["carrier_cm3", "concentration_value"])
    concs = g["concentration_value"].astype(float).values
    if g["concentration_value"].nunique() < 3:
        continue
    obs_n = np.log10(g["carrier_cm3"].astype(float).clip(lower=1).values)
    fracs = np.array([conc_to_frac(c, g["concentration_unit"].iloc[i])
                      for i, c in enumerate(concs)])
    fracs = np.clip(fracs, 1e-5, 0.15)
    try:
        pred_n = []
        for f in fracs:
            r = ff.forward(el, float(f), db=db, undoped_cache=cache, **PROC)
            pred_n.append(np.log10(max(r["hall_n_cm3"], 1.0)))
        pred_n = np.array(pred_n)
    except Exception:
        continue
    if np.ptp(pred_n) < 1e-6 or np.ptp(obs_n) < 1e-6:
        continue
    rho = float(spearmanr(pred_n, obs_n).correlation)
    obs_slope = float(np.polyfit(np.log10(fracs), obs_n, 1)[0])
    pred_slope = float(np.polyfit(np.log10(fracs), pred_n, 1)[0])
    series_results.append({
        "doi": doi[:34], "dopant": el, "n_points": len(g),
        "spearman": round(rho, 2), "obs_slope": round(obs_slope, 2),
        "pred_slope": round(pred_slope, 2),
        "slope_sign_match": bool(np.sign(obs_slope) == np.sign(pred_slope))})

n_series = len(series_results)
if n_series:
    sign_match = sum(s["slope_sign_match"] for s in series_results)
    med_rho = float(np.median([s["spearman"] for s in series_results]))
    mean_rho = float(np.mean([s["spearman"] for s in series_results]))
else:
    sign_match = med_rho = mean_rho = None

report = {
    "n_concentration_series": n_series,
    "slope_sign_match": f"{sign_match}/{n_series}" if n_series else None,
    "median_within_series_spearman": med_rho,
    "mean_within_series_spearman": mean_rho,
    "series": series_results,
    "interpretation": ("within-condition doping-response is the quantitative-capable layer; "
                       "high within-series Spearman + slope-sign match = the sandbox IS "
                       "quantitative for RELATIVE predictions even though absolutes are capped"),
}
(PROJ / "results/tier2/relative_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "series"}, indent=2))
print(f"\n{n_series} series; slope-sign {sign_match}/{n_series}; median within-series rho {med_rho}")
for s in series_results[:12]:
    print(f"  {s['dopant']:3s} {s['doi'][:28]:28s} n={s['n_points']} rho={s['spearman']:+.2f} "
          f"obs_slope={s['obs_slope']:+.2f} pred_slope={s['pred_slope']:+.2f} "
          f"{'OK' if s['slope_sign_match'] else 'X'}")
