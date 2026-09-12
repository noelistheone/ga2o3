"""
Phase 7A — Table 6 ("big combined table").

One row per dopant. Columns:
  dopant | N_int | PDR R²_raw | PDR R²_Platt | PDR r | PDR MAE | PDR N_ext | PDR ext r |
  VC R²_raw | VC R²_Platt | VC r | VC MAE
Rows ordered by N_int (PDR) descending.

Emits LaTeX booktabs + matching CSV at {out_dir}/tab_big_combined.{tex,csv}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


def _per_dopant(oof: pd.DataFrame, target: str):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in oof.columns:
        pred_pl = f"{target}_pred"
    pred_raw = f"{target}_pred"
    true = f"{target}_true"
    sub = oof.dropna(subset=[pred_pl, true])
    rows = {}
    for dop, g in sub.groupby("dopant_label"):
        yt = g[true].to_numpy()
        yp_pl = g[pred_pl].to_numpy()
        yp_raw = g[pred_raw].to_numpy() if pred_raw in g.columns else yp_pl
        if len(g) < 2 or np.std(yp_pl) < 1e-9:
            continue
        rows[dop] = dict(
            N=len(g),
            r2_raw=r2_score(yt, yp_raw),
            r2_platt=r2_score(yt, yp_pl),
            r=pearsonr(yt, yp_pl)[0],
            mae=mean_absolute_error(yt, yp_pl),
        )
    return rows


def _ext_per_dopant(ext: pd.DataFrame, target: str):
    pred_pl = f"{target}_pred_per_dopant_platt"
    if pred_pl not in ext.columns:
        pred_pl = f"{target}_pred"
    true = f"{target}_true"
    sub = ext.dropna(subset=[pred_pl, true])
    rows = {}
    for dop, g in sub.groupby("dopant_label"):
        if len(g) < 2 or np.std(g[pred_pl].to_numpy()) < 1e-9:
            rows[dop] = dict(N=len(g), r=float("nan"))
            continue
        yt = g[true].to_numpy()
        yp = g[pred_pl].to_numpy()
        rows[dop] = dict(
            N=len(g),
            r=pearsonr(yt, yp)[0],
        )
    return rows


def _fmt(v, sig=3):
    if pd.isna(v):
        return "---"
    return f"{v:+.{sig}f}"


def _build_frame(pdr_oof, pdr_ext, vc_oof, vc_ext):
    pdr_s = _per_dopant(pdr_oof, "photo_dark_ratio")
    pdr_e = _ext_per_dopant(pdr_ext, "photo_dark_ratio")
    vc_s = _per_dopant(vc_oof, "vacancy_concentration") if vc_oof is not None else {}
    vc_e = _ext_per_dopant(vc_ext, "vacancy_concentration") if vc_ext is not None else {}

    all_dop = sorted(set(pdr_s) | set(vc_s) | set(pdr_e) | set(vc_e),
                     key=lambda d: -(pdr_s.get(d, {}).get("N", 0)))
    rows = []
    for dop in all_dop:
        rows.append(dict(
            dopant=dop,
            N_int_pdr=pdr_s.get(dop, {}).get("N", 0),
            pdr_r2_raw=pdr_s.get(dop, {}).get("r2_raw", np.nan),
            pdr_r2_platt=pdr_s.get(dop, {}).get("r2_platt", np.nan),
            pdr_r=pdr_s.get(dop, {}).get("r", np.nan),
            pdr_mae=pdr_s.get(dop, {}).get("mae", np.nan),
            N_ext_pdr=pdr_e.get(dop, {}).get("N", 0),
            pdr_ext_r=pdr_e.get(dop, {}).get("r", np.nan),
            N_int_vc=vc_s.get(dop, {}).get("N", 0),
            vc_r2_raw=vc_s.get(dop, {}).get("r2_raw", np.nan),
            vc_r2_platt=vc_s.get(dop, {}).get("r2_platt", np.nan),
            vc_r=vc_s.get(dop, {}).get("r", np.nan),
            vc_mae=vc_s.get(dop, {}).get("mae", np.nan),
        ))
    return pd.DataFrame(rows)


def _to_latex(df: pd.DataFrame) -> str:
    head = r"""\begin{table*}[htbp]
\centering
\small
\caption{Per-dopant prediction quality for the single-element Phase 6 deployment
(PDR = 6D method/substrate-shuffle, VC = 6E within-DOI interpolation), with
per-dopant Platt calibration applied. External PDR is evaluated on the
literature set (N = 9 labelled rows); external VC is omitted pending a CSV bug
fix.}
\label{tab:big_combined}
\begin{tabular}{l r r r r r r r r r r r r}
\toprule
 & \multicolumn{6}{c}{Photo/dark ratio (PDR)} & \multicolumn{6}{c}{Vacancy concentration (VC)} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13}
Dopant & $N_{\text{int}}$ & $R^2_{\text{raw}}$ & $R^2_{\text{Platt}}$ & $r$ & MAE & ext $N$/$r$ &
         $N_{\text{int}}$ & $R^2_{\text{raw}}$ & $R^2_{\text{Platt}}$ & $r$ & MAE & ext $N$/$r$ \\
\midrule
"""
    lines = []
    for r in df.itertuples():
        ext_pdr = "---" if r.N_ext_pdr == 0 else (
            f"{int(r.N_ext_pdr)}/{_fmt(r.pdr_ext_r)}" if not pd.isna(r.pdr_ext_r) else f"{int(r.N_ext_pdr)}/---"
        )
        row = [
            r.dopant,
            str(int(r.N_int_pdr)),
            _fmt(r.pdr_r2_raw),
            _fmt(r.pdr_r2_platt),
            _fmt(r.pdr_r),
            "---" if pd.isna(r.pdr_mae) else f"{r.pdr_mae:.2f}",
            ext_pdr,
            str(int(r.N_int_vc)),
            _fmt(r.vc_r2_raw),
            _fmt(r.vc_r2_platt),
            _fmt(r.vc_r),
            "---" if pd.isna(r.vc_mae) else f"{r.vc_mae:.2f}",
            "---",   # VC ext column placeholder
        ]
        lines.append(" & ".join(row) + r" \\")
    foot = r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return head + "\n".join(lines) + "\n" + foot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-oof", type=Path, required=True)
    ap.add_argument("--pdr-ext", type=Path, required=True)
    ap.add_argument("--vc-oof", type=Path, default=None)
    ap.add_argument("--vc-ext", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    pdr_oof = pd.read_csv(args.pdr_oof)
    pdr_ext = pd.read_csv(args.pdr_ext)
    vc_oof = pd.read_csv(args.vc_oof) if args.vc_oof and args.vc_oof.exists() else None
    vc_ext = pd.read_csv(args.vc_ext) if args.vc_ext and args.vc_ext.exists() else None

    df = _build_frame(pdr_oof, pdr_ext, vc_oof, vc_ext)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "tab_big_combined.csv", index=False)
    (args.out_dir / "tab_big_combined.tex").write_text(_to_latex(df))
    print(f"Wrote tab_big_combined.tex + .csv to {args.out_dir}")


if __name__ == "__main__":
    main()
