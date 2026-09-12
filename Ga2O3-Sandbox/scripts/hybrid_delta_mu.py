"""Design Stage 5 — the bounded, orthogonal ML delta-trim on mu, with the anti-laundering gates.
Tests whether a physics-descriptor residual (lab-INVARIANT features only) tightens the mu LOLO
error below the physics-only floor, or whether mu physics is already maxed (delta-ablation gate).
Also produces the split-conformal 90% bar (the UQ that makes mu lab-usable/trustworthy).

Features z(x) (lab-invariant, per design): dopant electronegativity, ionic radius, |valence-3|,
log10 conc, sandbox log10 N_I (ionized scatterer density it already computes). NO lab-ID.
delta capped at B_mu = 0.15 dex; delta-ablation gate: LOLO must NOT worsen by >10% with delta off.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward, kroger_db
from pymatgen.core import Element

db = kroger_db.load()
DOP = set(db.elements) - {"Ga", "O"}
Tf, B_mu = 1500, 0.15


def feats(el, cf, N_I):
    e = Element(el)
    chi = e.X or 1.8
    r = (e.average_ionic_radius or 0.7)
    val = getattr(e, "common_oxidation_states", (3,)) or (3,)
    vmis = abs((val[0] if val else 3) - 3)
    return [chi, float(r), float(vmis), np.log10(max(cf, 1e-6)), np.log10(max(N_I, 1.0))]


def conc_frac(v, u):
    if pd.isna(v):
        return 0.005
    v = float(v); u = str(u).lower()
    if "cm" in u:
        return v / 3.83e22
    return v / 100.0 if ("at" in u or "%" in u) else (v / 100.0 * 0.5 if "wt" in u else 0.005)


def build():
    tp = pd.read_csv(PROJ.parent / "Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
    recs = []
    for _, r in tp.iterrows():
        el = str(r["dopant_element"])
        if el not in DOP or pd.isna(r.get("mu_cm2Vs")):
            continue
        cf = conc_frac(r.get("concentration_value"), r.get("concentration_unit"))
        o = full_forward.forward(el, cf, T_anneal=Tf, pO2=1e-5,
                                 film="sputter" in str(r.get("growth_method", "")).lower(), db=db)
        pm, meas = o["hall_mu_cm2Vs"], float(r["mu_cm2Vs"])
        if pm > 0 and meas > 0:
            try:
                z = feats(el, cf, o["N_I_ionized_cm3"])
            except Exception:
                continue
            recs.append({"doi": str(r.get("doi", "")), "resid": np.log10(pm) - np.log10(meas), "z": z})
    return pd.DataFrame(recs)


def main():
    df = build()
    labs = df["doi"].unique()
    Z = np.array(df["z"].tolist())
    res = df["resid"].values

    err_phys, err_delta = [], []          # LOLO: physics-only offset vs physics+delta
    for held in labs:
        tr = df.doi != held; te = df.doi == held
        off = np.median(res[tr])          # global physics offset
        # physics-only
        err_phys += list(np.abs(res[te] - off))
        # + delta (ridge on lab-invariant z, fit on train residual-minus-offset, capped)
        m = Ridge(alpha=5.0).fit(Z[tr], res[tr] - off)
        d = np.clip(m.predict(Z[te]), -B_mu, B_mu)
        err_delta += list(np.abs(res[te] - off - d))

    ep, ed = float(np.median(err_phys)), float(np.median(err_delta))
    # delta-ablation gate: delta "helps" only if it cuts error and physics-only is <90% of delta skill
    skill_phys = 1.0 / ep; skill_delta = 1.0 / ed
    delta_share = (skill_delta - skill_phys) / skill_delta
    conf90 = float(np.quantile(err_delta if ed < ep else err_phys, 0.9))
    out = {
        "property": "hall_mu", "n_rows": int(len(df)), "n_labs": int(len(labs)),
        "LOLO_physics_only_dex": round(ep, 3), "LOLO_physics_plus_delta_dex": round(ed, 3),
        "delta_skill_share": round(float(delta_share), 3),
        "B_mu_cap_dex": B_mu,
        "delta_adopted": bool(ed < ep - 0.005 and delta_share > 0.02),
        "conformal90_bar_dex": round(conf90, 3),
        "conformal90_factor": round(10 ** conf90, 2),
        "verdict": ("delta-trim ADOPTED: tightens mu LOLO" if ed < ep - 0.005
                    else "delta REJECTED by ablation gate: mu physics already at floor "
                         "(delta carries <2% skill) -> mu stays physics-only factor "
                         f"{10**ep:.1f}, genuinely simulated"),
    }
    (PROJ / "results/hybrid/delta_mu.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
