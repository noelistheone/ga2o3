"""
Phase 7A — Table 5: per-dopant within-DOI concentration-response quality.

For each (dopant, DOI) with ≥2 labelled rows on the same target, compute:
  - slope_true  = linear slope of log₁₀(target) vs log₁₀(concentration)
  - slope_pred  = same, using model predictions (per-dopant Platt calibrated)
  - rho_conc_pred = Pearson ρ(log₁₀(conc), pred) on those rows

Aggregate per dopant: median slope_true, median slope_pred, mean ρ, N_DOI.
Emit LaTeX booktabs + matching CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

MIN_ROWS_PER_DOI = 2


def _slope(x, y):
    if len(x) < 2 or np.std(x) < 1e-9:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _rho(x, y):
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(pearsonr(x, y)[0])


def _compute(df: pd.DataFrame, target: str) -> pd.DataFrame:
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in df.columns:
        pred_pl = f"{target}_pred"
    true = f"{target}_true"
    df = df.dropna(subset=[pred_pl, true, "concentration_at%", "doi", "element"])

    records = []
    for (doi, elem), g in df.groupby(["doi", "element"]):
        if len(g) < MIN_ROWS_PER_DOI:
            continue
        if str(elem) in ("—", "nan"):
            continue
        log_c = np.log10(np.clip(g["concentration_at%"].astype(float).to_numpy(), 1e-4, None))
        yt = g[true].to_numpy()
        yp = g[pred_pl].to_numpy()
        records.append(dict(
            doi=str(doi), element=str(elem), N=len(g),
            slope_true=_slope(log_c, yt),
            slope_pred=_slope(log_c, yp),
            rho_conc_pred=_rho(log_c, yp),
        ))
    return pd.DataFrame(records)


def _aggregate(per_doi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for elem, g in per_doi.groupby("element"):
        rows.append(dict(
            dopant=elem,
            N_DOI=len(g),
            median_slope_true=g["slope_true"].median(),
            median_slope_pred=g["slope_pred"].median(),
            mean_rho_conc_pred=g["rho_conc_pred"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("N_DOI", ascending=False).reset_index(drop=True)


def _to_latex(agg: pd.DataFrame, caption: str, label: str) -> str:
    header = r"""\begin{table}[htbp]
\centering
\small
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{l r r r r}
\toprule
Dopant & $N_{\text{DOI}}$ & median slope$_{\text{true}}$ & median slope$_{\text{pred}}$ & mean $\rho$(conc, pred) \\
\midrule
"""
    lines = []
    for r in agg.itertuples():
        def fmt(v):
            return "---" if pd.isna(v) else f"{v:+.3f}"
        lines.append(f"{r.dopant} & {r.N_DOI} & {fmt(r.median_slope_true)} & "
                     f"{fmt(r.median_slope_pred)} & {fmt(r.mean_rho_conc_pred)} \\\\")
    body = "\n".join(lines) + "\n"
    footer = r"""\bottomrule
\end{tabular}
\end{table}
"""
    return header + body + footer


def _load_merged(oof: Path, src: Path) -> pd.DataFrame:
    o = pd.read_csv(oof)
    s = pd.read_csv(src).reset_index().rename(columns={"index": "sample_idx"})
    keep = [c for c in ["sample_idx", "concentration_at%", "doi", "element"] if c in s.columns]
    return o.merge(s[keep], on="sample_idx", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-oof", type=Path, required=True)
    ap.add_argument("--pdr-src", type=Path, required=True)
    ap.add_argument("--vc-oof", type=Path, default=None)
    ap.add_argument("--vc-src", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdr_m = _load_merged(args.pdr_oof, args.pdr_src)
    pdr_per_doi = _compute(pdr_m, "photo_dark_ratio")
    pdr_agg = _aggregate(pdr_per_doi)
    pdr_agg.to_csv(args.out_dir / "tab5_concentration_slope_pdr.csv", index=False)
    (args.out_dir / "tab5_concentration_slope_pdr.tex").write_text(
        _to_latex(pdr_agg,
                  "Per-dopant within-DOI concentration response (PDR). Slopes are "
                  "regression coefficients of log$_{10}$(PDR) vs log$_{10}$(concentration); "
                  "$\\rho$ is Pearson correlation of log$_{10}$(conc) vs per-dopant-Platt "
                  "predictions on the same rows.",
                  "tab:conc_slope_pdr")
    )
    print(f"Wrote tab5 (PDR): {args.out_dir/'tab5_concentration_slope_pdr.tex'}")

    if args.vc_oof and args.vc_src:
        vc_m = _load_merged(args.vc_oof, args.vc_src)
        vc_per_doi = _compute(vc_m, "vacancy_concentration")
        vc_agg = _aggregate(vc_per_doi)
        vc_agg.to_csv(args.out_dir / "tab5_concentration_slope_vc.csv", index=False)
        (args.out_dir / "tab5_concentration_slope_vc.tex").write_text(
            _to_latex(vc_agg,
                      "Per-dopant within-DOI concentration response (VC).",
                      "tab:conc_slope_vc")
        )
        print(f"Wrote tab5 (VC): {args.out_dir/'tab5_concentration_slope_vc.tex'}")


if __name__ == "__main__":
    main()
