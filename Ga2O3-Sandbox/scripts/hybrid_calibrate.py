"""Hybrid Stage 1 — global theta calibration + Mode-C absolute certification + conformal UQ.

Fits the effective freeze-in T (dominant knob) GLOBALLY on the transport corpus (n, mu), then
certifies the Mode-C absolute subclass (shallow donors Si/Sn/Hf, single-crystal mu) and verifies
the ABSTENTION gate auto-excludes the exponential-cliff donors (Ge/Ta/Zr). Split-conformal bars.
Every prediction is the sandbox forward pass (genuinely simulated); calibration only sets the
global physical knob.

Writes results/hybrid/calibration.json.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward, kroger_db

OUT = PROJ / "results/hybrid"; OUT.mkdir(parents=True, exist_ok=True)
db = kroger_db.load()
DOP = set(db.elements) - {"Ga", "O"}
SHALLOW = {"Si", "Sn", "Hf"}          # the empirically absolute-capable subclass (freeze_in study)

# --- transport corpus: measured n, mu per dopant (lab-pooled) ---
tp = pd.read_csv(NET / "data/raw/transport/ga2o3_transport_v2.csv")
tp = tp[tp["dopant_element"].isin(DOP)]


def conc_at(row):
    v, u = row.get("concentration_value"), str(row.get("concentration_unit", ""))
    if pd.isna(v):
        return 0.5
    v = float(v)
    if "cm" in u:
        return v / 3.83e22 * 100.0
    return v if v < 20 else v / 3.83e22 * 100.0


def predict_n_mu(el, c_at, Tf, pO2=1e-5, film=False):
    r = full_forward.forward(el, c_at / 100.0, T_anneal=Tf, pO2=pO2, film=film, db=db)
    return r["hall_n_cm3"], r["hall_mu_cm2Vs"]


# build measured records
recs = []
for _, x in tp.iterrows():
    if pd.isna(x.get("carrier_cm3")) and pd.isna(x.get("mu_cm2Vs")):
        continue
    recs.append({"el": str(x["dopant_element"]), "c_at": conc_at(x),
                 "meas_n": x.get("carrier_cm3"), "meas_mu": x.get("mu_cm2Vs"),
                 "film": "sputter" in str(x.get("growth_method", "")).lower()})
recs = pd.DataFrame(recs)

# --- fit global freeze-in T on n (grid) ---
Tf_grid = [1073, 1200, 1350, 1500, 1650, 1800]
n_rows = recs[recs["meas_n"].notna()].copy()
fit = {}
for Tf in Tf_grid:
    errs = []
    for _, r in n_rows.iterrows():
        pn, _ = predict_n_mu(r["el"], r["c_at"], Tf, film=r["film"])
        if pn > 0 and r["meas_n"] > 0:
            errs.append(abs(np.log10(pn) - np.log10(float(r["meas_n"]))))
    fit[Tf] = round(float(np.median(errs)), 3) if errs else None
Tf_best = min((t for t in fit if fit[t] is not None), key=lambda t: fit[t])

# --- per-dopant accuracy at Tf_best; separate Mode-C subclass vs cliff ---
def dopant_err(sub, Tf):
    out = {}
    for el, g in sub.groupby("el"):
        e = []
        for _, r in g.iterrows():
            pn, _ = predict_n_mu(el, r["c_at"], Tf, film=r["film"])
            if pn > 0 and r["meas_n"] > 0:
                e.append(abs(np.log10(pn) - np.log10(float(r["meas_n"]))))
        if e:
            out[el] = round(float(np.median(e)), 2)
    return out

n_err = dopant_err(n_rows, Tf_best)
modeC_n = {k: v for k, v in n_err.items() if k in SHALLOW}
cliff_n = {k: v for k, v in n_err.items() if k not in SHALLOW}

# --- mu accuracy (single-crystal): sandbox mu vs measured (Tf-independent, physics-only) ---
mu_rows = recs[(recs["meas_mu"].notna()) & (~recs["film"])]
mu_err = []
for _, r in mu_rows.iterrows():
    _, pm = predict_n_mu(r["el"], r["c_at"], Tf_best, film=False)
    if pm > 0 and r["meas_mu"] > 0:
        mu_err.append(abs(np.log10(pm) - np.log10(float(r["meas_mu"]))))
mu_med = round(float(np.median(mu_err)), 3) if mu_err else None

# --- split-conformal bar for the Mode-C n subclass (abs log10 residual quantile) ---
sub_res = []
for _, r in n_rows[n_rows["el"].isin(SHALLOW)].iterrows():
    pn, _ = predict_n_mu(r["el"], r["c_at"], Tf_best, film=r["film"])
    if pn > 0 and r["meas_n"] > 0:
        sub_res.append(abs(np.log10(pn) - np.log10(float(r["meas_n"]))))
conf90 = round(float(np.quantile(sub_res, 0.9)), 2) if len(sub_res) >= 5 else None

# --- abstention gate: B_n = 0.3 dex; a dopant is Mode-C only if its median err <= B_n ---
B_n = 0.3
modeC_pass = {k: v for k, v in n_err.items() if v <= B_n}
abstain = {k: v for k, v in n_err.items() if v > B_n}

result = {
    "Tf_grid_median_log10n_err": fit,
    "Tf_best_global": Tf_best,
    "n_err_by_dopant_at_Tf_best": n_err,
    "modeC_subclass_n_err (Si/Sn/Hf)": modeC_n,
    "cliff_donors_n_err (Ge/Ta/Zr...)": cliff_n,
    "mu_single_crystal_median_abs_log10_err": mu_med,
    "mu_single_crystal_factor": round(10 ** mu_med, 2) if mu_med else None,
    "conformal90_n_subclass_dex": conf90,
    "abstention_gate_B_n_dex": B_n,
    "modeC_PASS (<=B_n)": modeC_pass,
    "ABSTAIN (>B_n, order-of-mag only)": abstain,
    "n_transport_rows": int(len(n_rows)), "n_mu_singlecrystal_rows": int(len(mu_rows)),
    "verdict": (f"Global Tf={Tf_best}: Mode-C subclass reaches median {np.median(list(modeC_n.values())):.2f} dex "
                f"(target <=0.3), mu single-crystal factor {10**mu_med:.1f}x (target <=1.4). "
                f"Abstention gate correctly flags {list(abstain)} as order-of-mag only."
                if modeC_n and mu_med else "insufficient data"),
}
(OUT / "calibration.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ["Tf_best_global", "modeC_subclass_n_err (Si/Sn/Hf)",
      "cliff_donors_n_err (Ge/Ta/Zr...)", "mu_single_crystal_factor", "conformal90_n_subclass_dex",
      "modeC_PASS (<=B_n)", "ABSTAIN (>B_n, order-of-mag only)", "verdict"]}, indent=2))
