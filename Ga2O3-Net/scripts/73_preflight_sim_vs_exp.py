"""Phase 61 PRE-FLIGHT — does the closed-form physics simulator track experimental [V_O]?

The decisive, on-disk gate (research Brief 4 / Kennedy-O'Hagan): for a differentiable
defect-equilibrium-solver + Δ-correction model to break the magnitude ceiling, the
*raw simulator* prediction must already CORRELATE with the experimental target. If it
does, the residual (exp - sim) is small/smooth and a tiny Δ-head can learn it at N≈14.
If it does NOT, no statistical correction can manufacture magnitude signal — the honest
answer is "magnitude needs new experiments," and the contribution is by-construction
physics + honest uncertainty.

Computes (sim, exp, doi, element) per VC row using the project's OWN closed-form solver
(src/data/dcwm_transitions.measured_log_vo + kroger_synthetic.kroger_predict + the n-type
Boltzmann-sum) and the project's OWN CSV parsing. Reports:
  - raw Pearson/Spearman/R² (sim vs exp), scale/offset-free correlation is the K-O'Hagan check
  - GroupKFold-by-DOI OOF R² of a GLOBAL AFFINE on the simulator (exp ~ a*sim + b) — the fair,
    leakage-free "can a single rescale of the simulator predict a held-out paper?" number
  - residual std vs target std (is the correction small & smooth?)
  - per-element and per-DOI correlations (the per-element axis)
All metrics saved to results/phase61_preflight/sim_vs_exp.json (workdir, never /tmp).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sys
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.dcwm_transitions import (  # noqa: E402
    measured_log_vo, bulk_log_vo, _parse_conc_frac, atmosphere_to_key,
    method_to_idx, ATM_PO2, LOG_PREFACTOR, KB_EV, LN10, C_REF,
)
from src.data.kroger_synthetic import kroger_predict, DOPANT_OFFSET  # noqa: E402

CSV = PROJ / "data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v56.csv"
OUT = PROJ / "results/phase61_preflight/sim_vs_exp.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# n-type-Fermi Boltzmann-sum closed form (scripts/41_v58_kroger_to_qa convention; eps_F near CBM)
NATIVE_VO_EF_NTYPE = {0: 3.5, 1: 1.8 - 1.0 * 1.0, 2: 0.3 - 2.0 * 1.0}  # approx n-type shift, illustrative


def _ntype_total(elem, c, T_K, log_pO2):
    """Boltzmann sum over q=0,1,2 with an n-type Fermi level (~CBM); dopant offset + atmosphere."""
    offset = DOPANT_OFFSET.get(elem, 0.0)
    dop = offset * math.log10(max(c, 1e-6) / C_REF) if c and c > 0 else 0.0
    # n-type eps_F = 4.0 eV (near CBM, EG=4.85); E_f(q) = E_f0(q) - q*eps_F (midgap->ntype shift)
    epsF = 4.0
    base = {0: 3.5, 1: 1.8, 2: 0.3}  # midgap HSE06 (Lyons)
    logs = []
    for q in (0, 1, 2):
        ef = base[q] - q * (epsF - 4.85 / 2.0)  # shift from midgap to n-type
        logs.append(LOG_PREFACTOR - ef / (KB_EV * T_K * LN10) - 0.5 * log_pO2 + dop)
    logs = np.array(logs)
    m = logs.max()
    total = m + math.log10(float(np.sum(10.0 ** (logs - m))))
    return min(total, LOG_PREFACTOR)


def _r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return _pearson(ra, rb)


def main():
    df = pd.read_csv(CSV)
    y = pd.to_numeric(df["vacancy_concentration"], errors="coerce")
    df = df[y.notna()].reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        elem = str(row.get("element") or "undoped").strip()
        if elem in ("—", "-", "nan", ""):
            elem = "undoped"
        c = _parse_conc_frac(row)
        T_raw = pd.to_numeric(row.get("temperature_C"), errors="coerce")
        T_C = float(T_raw) if pd.notna(T_raw) else 25.0
        T_K = T_C + 273.15
        atm = atmosphere_to_key(str(row.get("atmosphere")))
        log_pO2 = math.log10(ATM_PO2[atm])
        cc = c if c > 0 else 1e-6
        exp = float(pd.to_numeric(row["vacancy_concentration"], errors="coerce"))
        notes = str(row.get("notes") or "").lower()
        is_aug = ("interp" in notes) or ("augment" in notes) or ("aug" in notes)
        rows.append(dict(
            elem=elem, doi=str(row.get("doi")), method=method_to_idx(str(row.get("method"))),
            exp=exp, is_aug=is_aug,
            sim_measured=measured_log_vo(elem, cc, T_K, log_pO2),
            sim_bulk=bulk_log_vo(elem, cc, T_K, log_pO2),
            sim_kroger=kroger_predict(elem, cc, T_K, log_pO2, q=0),
            sim_ntype=_ntype_total(elem, cc, T_K, log_pO2),
        ))
    R = pd.DataFrame(rows)

    def evaluate(sub: pd.DataFrame, simcol: str) -> dict:
        sim = sub[simcol].to_numpy(float)
        exp = sub["exp"].to_numpy(float)
        out = dict(n=int(len(sub)), pearson=_pearson(sim, exp), spearman=_spearman(sim, exp),
                   r2_raw=_r2(exp, sim), std_exp=float(exp.std()),
                   std_resid_raw=float((exp - sim).std()))
        # global affine fit (in-sample) — best a*sim+b
        if sim.std() > 0:
            a, b = np.polyfit(sim, exp, 1)
            out["affine_a"], out["affine_b"] = float(a), float(b)
            out["r2_affine_insample"] = _r2(exp, a * sim + b)
            out["std_resid_affine"] = float((exp - (a * sim + b)).std())
        # GroupKFold-by-DOI OOF: fit affine on other DOIs, predict held-out DOI
        dois = sub["doi"].to_numpy()
        uniq = list(dict.fromkeys(dois))
        oof = np.full(len(sub), np.nan)
        for d in uniq:
            tr = dois != d
            te = dois == d
            if tr.sum() >= 3 and sim[tr].std() > 0:
                a, b = np.polyfit(sim[tr], exp[tr], 1)
            else:
                a, b = 0.0, exp[tr].mean() if tr.sum() else exp.mean()
            oof[te] = a * sim[te] + b
        out["r2_affine_oof_gkf"] = _r2(exp, oof)
        out["n_doi"] = len(uniq)
        # null: predict global mean (R²=0 by def) and per-DOI mean (oracle upper bound on between-DOI)
        return out

    report = {"csv": str(CSV), "n_vc_rows": int(len(R)), "n_original_nonaug": int((~R["is_aug"]).sum())}
    for simcol in ["sim_measured", "sim_kroger", "sim_ntype", "sim_bulk"]:
        report[simcol] = {
            "all_rows": evaluate(R, simcol),
            "original_nonaug": evaluate(R[~R["is_aug"]], simcol) if (~R["is_aug"]).sum() >= 4 else None,
        }
    # per-element correlation (measured sim) — the per-element axis
    pe = {}
    for el, g in R.groupby("elem"):
        if len(g) >= 4:
            pe[el] = dict(n=int(len(g)), pearson=_pearson(g["sim_measured"], g["exp"]),
                          spearman=_spearman(g["sim_measured"], g["exp"]))
    report["per_element_sim_measured"] = pe
    # per-DOI within-paper correlation (does sim track the within-paper trend?)
    pd_ = {}
    for d, g in R.groupby("doi"):
        if len(g) >= 3 and g["sim_measured"].std() > 0:
            pd_[d] = dict(n=int(len(g)), pearson=_pearson(g["sim_measured"], g["exp"]))
    report["per_doi_within_sim_measured"] = pd_
    report["within_doi_mean_pearson"] = float(np.nanmean([v["pearson"] for v in pd_.values()])) if pd_ else None

    OUT.write_text(json.dumps(report, indent=2))
    # console summary
    print(f"VC rows: {report['n_vc_rows']}  (original non-aug: {report['n_original_nonaug']})  "
          f"DOIs: {report['sim_measured']['all_rows']['n_doi']}")
    print("\n=== RAW SIMULATOR vs EXPERIMENT (scale/offset-free Pearson is the K-O'Hagan check) ===")
    for sc in ["sim_measured", "sim_kroger", "sim_ntype", "sim_bulk"]:
        a = report[sc]["all_rows"]
        print(f"  {sc:13s} | all N={a['n']:3d}  Pearson={a['pearson']:+.3f}  Spearman={a['spearman']:+.3f}"
              f"  R2_raw={a['r2_raw']:+.2f}  R2_affine_OOF_GKF={a['r2_affine_oof_gkf']:+.3f}"
              f"  resid_std={a['std_resid_raw']:.2f}/{a['std_exp']:.2f}(exp)")
        o = report[sc]["original_nonaug"]
        if o:
            print(f"  {'':13s} | orig N={o['n']:3d}  Pearson={o['pearson']:+.3f}"
                  f"  R2_affine_OOF_GKF={o['r2_affine_oof_gkf']:+.3f}")
    print(f"\nwithin-DOI mean Pearson (sim_measured tracks within-paper trend): "
          f"{report['within_doi_mean_pearson']}")
    print("\nper-element (sim_measured):")
    for el, v in report["per_element_sim_measured"].items():
        print(f"  {el:8s} N={v['n']:3d}  Pearson={v['pearson']:+.3f}  Spearman={v['spearman']:+.3f}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
