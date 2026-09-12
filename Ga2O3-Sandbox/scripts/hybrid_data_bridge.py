"""Hybrid data bridge: run the SANDBOX forward simulator on every experimental corpus row and
pair the (genuinely-simulated) prediction with the measured value + the lab id (doi) — the
foundation for the ML-sandbox hybrid (calibration + residual with the lab confound quarantined).

Also directly quantifies the sandbox's CURRENT absolute accuracy on the full corpus (not just
2 series) — an audit deliverable.

Corpus (Ga2O3-Net, read-only): ga2o3_exp.csv (PDR, dark, [V_O], device) + transport_v2.csv
(hall n, mu, sigma). Maps element->dopant, concentration_at%->cation fraction, atmosphere->pO2,
temperature_C->T_anneal, method->film. Runs on the KROGER-19 dopants (sandbox-native); new
dopants (Sb/Bi/...) use the Tier-3 injection separately.

Writes results/hybrid/bridge_data.csv (one row per (doi, element, conc, process) with pred_* and meas_*).
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sandbox import full_forward, kroger_db

NET = Path("/home/lawrence/Physics/Ga2O3-Net")
OUT = Path("/home/lawrence/Physics/Ga2O3-Sandbox/results/hybrid")
OUT.mkdir(parents=True, exist_ok=True)

KROGER19 = set(kroger_db.ELEMENTS)          # Ga O Si H Fe Sn Cr Ti Ir Mg Ca Zn Co Zr Hf Ta Ge Pt Rh
DOPANTS = KROGER19 - {"Ga", "O"}

PO2 = {"vacuum": 1e-8, "Ar": 1e-5, "inert": 1e-5, "N2": 1e-5, "Ar+N2": 1e-5,
       "air": 0.21, "O2": 1.0, "O2_plasma": 1.0, "N2+O2": 0.1, "Ar+O2": 0.1,
       "Ar:O2=1:1": 0.5}
FILM_METHODS = ("sputter", "pld", "sol", "spray", "cvd_film", "ald", "evaporation")


def po2_of(atm):
    if not isinstance(atm, str):
        return 1e-5
    return PO2.get(atm.strip(), 0.1 if "O2" in atm else 1e-5)


def is_film(method):
    if not isinstance(method, str):
        return True
    m = method.lower()
    if any(k in m for k in ("mocvd", "hvpe", "czochralski", "efg", "float", "bulk", "single")):
        return False
    return True


def run_forward(element, conc_at, atm, T_C, method, db):
    conc_frac = float(conc_at) / 100.0 if conc_at and conc_at > 0 else 0.0
    T_anneal = float(T_C) + 273.15 if T_C and not np.isnan(T_C) else 1073.0
    try:
        r = full_forward.forward(element, conc_frac, T_anneal=T_anneal, pO2=po2_of(atm),
                                 film=is_film(method), db=db)
        return r
    except Exception:
        return None


def main():
    db = kroger_db.load()
    rows = []

    # --- device/optical corpus: PDR, dark, [V_O] ---
    exp = pd.read_csv(NET / "data/raw/experimental/ga2o3_exp.csv")
    for _, x in exp.iterrows():
        el = str(x.get("element", "")).strip()
        if el not in DOPANTS:
            continue
        conc = x.get("concentration_at%")
        if pd.isna(conc):
            continue
        r = run_forward(el, conc, x.get("atmosphere"), x.get("temperature_C"), x.get("method"), db)
        if r is None:
            continue
        rows.append({
            "doi": x.get("doi"), "element": el, "conc_at": float(conc),
            "atmosphere": x.get("atmosphere"), "T_C": x.get("temperature_C"), "method": x.get("method"),
            "source": "exp",
            "pred_PDR": r["PDR_score"], "meas_PDR": x.get("photo_dark_ratio"),
            "pred_dark": r["dark_activation_eV"], "meas_dark": x.get("dark_current_pA"),
            "pred_VO": r["V_O_cm3_quench"], "meas_VO": x.get("vacancy_concentration"),
            "pred_n": r["hall_n_cm3"], "pred_mu": r["hall_mu_cm2Vs"], "pred_sigma": r["sigma_S_cm"],
            "pred_Eg": r["Eg_optical_eV"], "pred_dEg": r["dEg_eV"],
        })

    # --- transport corpus: hall n, mu, sigma ---
    tr = pd.read_csv(NET / "data/raw/transport/ga2o3_transport_v2.csv")
    for _, x in tr.iterrows():
        el = str(x.get("dopant_element", "")).strip()
        if el not in DOPANTS:
            continue
        cval, cunit = x.get("concentration_value"), str(x.get("concentration_unit", ""))
        # normalize concentration to at% (rough: cm-3 -> at% via cation density 3.83e22)
        conc_at = None
        if pd.notna(cval):
            if "at" in cunit or "%" in cunit:
                conc_at = float(cval)
            elif "cm" in cunit or "e" in str(cval).lower():
                conc_at = float(cval) / 3.83e22 * 100.0
        if conc_at is None or conc_at <= 0:
            conc_at = 0.5
        r = run_forward(el, conc_at, None, None, x.get("growth_method"), db)
        if r is None:
            continue
        rows.append({
            "doi": x.get("doi"), "element": el, "conc_at": conc_at,
            "atmosphere": None, "T_C": None, "method": x.get("growth_method"), "source": "transport",
            "pred_n": r["hall_n_cm3"], "meas_n": x.get("carrier_cm3"),
            "pred_mu": r["hall_mu_cm2Vs"], "meas_mu": x.get("mu_cm2Vs"),
            "pred_sigma": r["sigma_S_cm"], "meas_sigma": (1.0 / float(x["rho_ohmcm"]) if pd.notna(x.get("rho_ohmcm")) else None),
            "pred_PDR": r["PDR_score"], "pred_dark": r["dark_activation_eV"],
            "pred_Eg": r["Eg_optical_eV"], "pred_dEg": r["dEg_eV"], "pred_VO": r["V_O_cm3_quench"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bridge_data.csv", index=False)
    print(f"bridge: {len(df)} rows, {df['doi'].nunique()} labs, {df['element'].nunique()} dopants")

    # quick current-sandbox absolute accuracy (log-space where applicable)
    def acc(pred, meas, logsp=True):
        m = df[[pred, meas]].dropna()
        if len(m) < 5:
            return None
        p, o = m[pred].astype(float).values, m[meas].astype(float).values
        if logsp:
            ok = (p > 0) & (o > 0)
            if ok.sum() < 5:
                return None
            err = np.abs(np.log10(p[ok]) - np.log10(o[ok]))
            return {"n": int(ok.sum()), "median_abs_log10_err": round(float(np.median(err)), 2),
                    "spearman": round(float(pd.Series(p[ok]).corr(pd.Series(o[ok]), method="spearman")), 3)}
        err = np.abs(p - o)
        return {"n": len(m), "median_abs_err": round(float(np.median(err)), 3),
                "spearman": round(float(pd.Series(p).corr(pd.Series(o), method="spearman")), 3)}

    print("\nCURRENT SANDBOX absolute accuracy on the corpus (uncalibrated):")
    for pred, meas, log in [("pred_n", "meas_n", True), ("pred_mu", "meas_mu", True),
                            ("pred_sigma", "meas_sigma", True), ("pred_PDR", "meas_PDR", True),
                            ("pred_dark", "meas_dark", True), ("pred_VO", "meas_VO", True)]:
        a = acc(pred, meas, log)
        if a:
            print(f"  {meas}: {a}")
    import json
    (OUT / "bridge_accuracy.json").write_text(json.dumps({
        p: acc(p, m, l) for p, m, l in [("pred_n", "meas_n", True), ("pred_mu", "meas_mu", True),
        ("pred_sigma", "meas_sigma", True), ("pred_PDR", "meas_PDR", True),
        ("pred_dark", "meas_dark", True), ("pred_VO", "meas_VO", True)]}, indent=2))


if __name__ == "__main__":
    main()
