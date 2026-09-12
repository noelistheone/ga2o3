"""R8 TA-07 (blocking): is the Ta 'calibrated reproduction' circular?
The certified engine's global freeze-in temperature (1500 K) was fit on row-level transport
data that includes Ta rows. Here: (1) refit the identical Tf grid with ALL Ta rows excluded;
(2) re-run the OFZ Ta loading series at the leave-Ta-out temperature and compare with the
published record (n 3.6e16-3e19 controllable, mu 138 at 7e17); (3) state the effective-pO2
selection rule and the n(loading) sensitivity across the OFZ state bracket already on file.
Writes results/tier2/r8_ta07_leave_ta_out.json.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()
DOP = set(db.elements) - {"Ga", "O"}
tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
tp = tp[tp["dopant_element"].isin(DOP)]


def conc_at(row):
    v = row.get("concentration_value")
    u = str(row.get("concentration_unit", "")).lower()
    if pd.isna(v):
        return 0.1
    v = float(v)
    if "at" in u or "%" in u:
        return v
    if "cm-3" in u or "cm^-3" in u or "cm3" in u:
        return 100.0 * v / 3.8e22
    return v


recs = []
for _, x in tp.iterrows():
    if pd.isna(x.get("carrier_cm3")):
        continue
    recs.append({"el": str(x["dopant_element"]), "c_at": conc_at(x),
                 "meas_n": float(x["carrier_cm3"]),
                 "film": "sputter" in str(x.get("growth_method", "")).lower()})
recs = pd.DataFrame(recs)
recs = recs[recs.meas_n > 0]

Tf_grid = [1073, 1200, 1350, 1500, 1650, 1800]


def grid_fit(sub):
    fit = {}
    for Tf in Tf_grid:
        errs = []
        for _, r in sub.iterrows():
            fr = ff.forward(r["el"], r["c_at"] / 100.0, T_anneal=float(Tf), pO2=1e-5,
                            film=bool(r["film"]), db=db)
            pn = fr["hall_n_cm3"]
            if pn > 0:
                errs.append(abs(np.log10(pn) - np.log10(r["meas_n"])))
        fit[Tf] = round(float(np.median(errs)), 3) if errs else None
    best = min((t for t in fit if fit[t] is not None), key=lambda t: fit[t])
    return fit, best


n_ta = int((recs.el == "Ta").sum())
fit_all, best_all = grid_fit(recs)
fit_noTa, best_noTa = grid_fit(recs[recs.el != "Ta"])
print(f"Ta rows in fit set: {n_ta}; Tf_best all={best_all}, leave-Ta-out={best_noTa}", flush=True)

# OFZ Ta series at the leave-Ta-out temperature (identical protocol to ta_experimental_conditions)
N_cat = kroger_db.N_SITE * 2 / 3
series = []
for f in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 2e-3):
    r = ff.forward("Ta", f, T_anneal=float(best_noTa), pO2=0.2, film=False, Nd_bg=1e16, db=db)
    series.append({"frac": f, "Ta_cm3": float(f * N_cat), "n_cm3": float(r["hall_n_cm3"]),
                   "mu_cm2Vs": round(float(r["hall_mu_cm2Vs"]), 1)})
    print(f"leave-Ta-out OFZ: frac={f:g} Ta={f*N_cat:.2e} n={r['hall_n_cm3']:.2e} "
          f"mu={r['hall_mu_cm2Vs']:.1f}", flush=True)

# mu at the substrate condition (record: 138 at n=7e17)
mu_at_7e17 = None
for s in series:
    if s["n_cm3"] > 0 and abs(np.log10(max(s["n_cm3"], 1.0)) - np.log10(7e17)) < 0.5:
        mu_at_7e17 = s["mu_cm2Vs"]

out = {"ta_rows_in_global_fit": n_ta,
       "Tf_grid_all": fit_all, "Tf_best_all": best_all,
       "Tf_grid_leave_ta_out": fit_noTa, "Tf_best_leave_ta_out": best_noTa,
       "ofz_series_at_leave_ta_out_Tf": series,
       "mu_near_7e17": mu_at_7e17,
       "effective_pO2_rule": "OFZ series run at air pO2=0.2 atm exactly as the published record "
                             "(growth ambient), not a tuned effective value; the effective-state "
                             "bracket (air vs O2-poor) is the pre-existing file "
                             "results/tier2/ta_experimental_conditions.json",
       "read": ("leave-Ta-out temperature equals the all-rows temperature -> the Ta reproduction "
                "is not circular" if best_noTa == best_all else
                "leave-Ta-out temperature differs -> reproduction re-evaluated at the clean "
                "temperature above")}
(PROJ / "results/tier2/r8_ta07_leave_ta_out.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_ta07_leave_ta_out.json")
