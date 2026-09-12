"""
compare_specialist_vs_frontier.py — head-to-head evaluation of element
specialists vs the universal frontier on the same element subset.

For Mg-VC: compare frontier (Phase 33) restricted to Mg-OOF rows against
the Mg-only specialist (Phase 39 Mg).
For Sn-PDR: same idea with Phase 27 vs Phase 39 Sn.

Reports (each on the element-subset of OOF):
  N, Pearson r, R²_Platt, MAE, RMSE, PI80 coverage, PI80 mean width.
PI80 uses the calibrated conformal multiplier (per-element-tier-0 if
available, else global) loaded from
results/lab_regime_eval/conformal_quantiles.json (run Task F first).

Routing recommendation per element is printed at the end:
  - Use SPECIALIST if Δr ≥ +0.02 or ΔR² ≥ +0.02
  - Use FRONTIER  otherwise (specialist failed to add value).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

PI80_Z = 1.2815515655446004
DEPLOY = PROJ / "results" / "deployment"

DEFAULT_PAIRS = [
    {
        "element": "Mg",
        "target":  "VC",
        "target_col": "vacancy_concentration",
        "frontier_dir":   DEPLOY / "vc_alt_33_sputter_brouwer_atmo_class",
        "specialist_dir": PROJ / "results" / "phase39_mg_vc_specialist_3seed",
    },
    {
        "element": "Sn",
        "target":  "PDR",
        "target_col": "photo_dark_ratio",
        "frontier_dir":   DEPLOY / "pdr_alt_27_sputter_cmixup",
        "specialist_dir": PROJ / "results" / "phase39_sn_pdr_specialist_3seed",
    },
]


def _conformal_q(target: str, element: str) -> tuple[float, str]:
    """Return calibrated multiplier for (target, element); fall back to global."""
    p = PROJ / "results" / "lab_regime_eval" / "conformal_quantiles.json"
    if not p.exists():
        return PI80_Z, "naive"
    blk = next(t for t in json.loads(p.read_text())["targets"] if t["target"] == target)
    qd = blk["per_elem_tier0_q"].get(element, {})
    if qd.get("q_norm") is not None:
        return float(qd["q_norm"]), f"conformal_local_{element}"
    if blk["per_tier_q"].get("0", {}).get("q_norm") is not None:
        return float(blk["per_tier_q"]["0"]["q_norm"]), "conformal_tier0"
    return float(blk["q_global_norm"]), "conformal_global"


def _metrics(df: pd.DataFrame, target_col: str, q_mult: float) -> dict:
    pred = df[f"{target_col}_pred_platt"].to_numpy()
    true = df[f"{target_col}_true"].to_numpy()
    std  = df[f"{target_col}_std"].to_numpy()
    mask = np.isfinite(pred) & np.isfinite(true) & np.isfinite(std)
    pred, true, std = pred[mask], true[mask], std[mask]
    n = len(true)
    if n < 2:
        return dict(n=n, r=float("nan"), r2_platt=float("nan"),
                    mae=float("nan"), rmse=float("nan"),
                    pi80_cov=float("nan"), pi80_width=float("nan"))
    r = float(stats.pearsonr(true, pred).statistic) if np.std(true) > 1e-9 else float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    half = q_mult * np.maximum(std, 1e-9)
    cov = float(np.mean(np.abs(true - pred) <= half))
    width = float(np.mean(2.0 * half))
    return dict(n=n, r=r, r2_platt=r2, mae=mae, rmse=rmse,
                pi80_cov=cov, pi80_width=width)


def evaluate_pair(pair: dict) -> dict:
    elem  = pair["element"]
    tgt   = pair["target"]
    tcol  = pair["target_col"]

    fr_path = pair["frontier_dir"]   / "oof_predictions.csv"
    sp_path = pair["specialist_dir"] / "oof_predictions.csv"
    if not fr_path.exists():
        raise FileNotFoundError(fr_path)
    if not sp_path.exists():
        raise FileNotFoundError(f"specialist not yet trained: {sp_path}")

    fr = pd.read_csv(fr_path)
    sp = pd.read_csv(sp_path)

    # Element subset (specialist OOF still includes undoped — keep only the
    # element of interest for a fair comparison).
    fr_e = fr[fr["dopant_label"] == elem].copy()
    sp_e = sp[sp["dopant_label"] == elem].copy()

    q_mult, q_kind = _conformal_q(tgt, elem)

    fr_m = _metrics(fr_e, tcol, q_mult)
    sp_m = _metrics(sp_e, tcol, q_mult)
    delta = {k: (sp_m[k] - fr_m[k]) if isinstance(sp_m[k], float)
                                       and isinstance(fr_m[k], float)
                                       and np.isfinite(sp_m[k]) and np.isfinite(fr_m[k])
                                    else float("nan")
             for k in fr_m if k != "n"}

    routing = ("specialist"
               if (delta.get("r", 0) >= 0.02 or delta.get("r2_platt", 0) >= 0.02)
               else "frontier")

    return dict(
        element=elem, target=tgt,
        q_mult=q_mult, q_kind=q_kind,
        frontier=fr_m, specialist=sp_m, delta=delta,
        routing=routing,
        frontier_dir=str(pair["frontier_dir"].relative_to(PROJ)),
        specialist_dir=str(pair["specialist_dir"].relative_to(PROJ)),
    )


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def write_report(blocks: list[dict], path: Path) -> None:
    L = ["# Element-Specialist vs Frontier — Head-to-Head", ""]
    L.append("Calibration: PI80 uses element-local conformal q where available; "
             "naive z=1.282 only if conformal_quantiles.json is missing.")
    L.append("")
    for b in blocks:
        L.append(f"## {b['element']}-{b['target']}  "
                 f"(PI multiplier source: {b['q_kind']}, q={b['q_mult']:.3f})")
        L.append("")
        L.append("| metric | frontier | specialist | Δ (specialist − frontier) |")
        L.append("|---|---:|---:|---:|")
        for key, label in [("n", "N"), ("r", "Pearson r"),
                           ("r2_platt", "R² (Platt)"), ("mae", "MAE"),
                           ("rmse", "RMSE"),
                           ("pi80_cov", "PI80 coverage"),
                           ("pi80_width", "PI80 mean width")]:
            f_val = b["frontier"][key]
            s_val = b["specialist"][key]
            if key == "n":
                L.append(f"| {label} | {f_val} | {s_val} | — |")
            else:
                L.append(f"| {label} | {fmt(f_val)} | {fmt(s_val)} | "
                         f"{fmt(b['delta'].get(key))} |")
        L.append("")
        L.append(f"**Routing recommendation for {b['element']}: "
                 f"`{b['routing']}`**")
        L.append("")
        L.append(f"frontier: `{b['frontier_dir']}`")
        L.append(f"specialist: `{b['specialist_dir']}`")
        L.append("")
    path.write_text("\n".join(L))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=PROJ / "results" / "lab_regime_eval")
    args = parser.parse_args()

    blocks = []
    for pair in DEFAULT_PAIRS:
        try:
            b = evaluate_pair(pair)
        except FileNotFoundError as e:
            print(f"[SKIP] {pair['element']}-{pair['target']}: {e}")
            continue
        blocks.append(b)
        print(f"\n=== {b['element']}-{b['target']}  "
              f"q={b['q_mult']:.3f} ({b['q_kind']}) ===")
        for label, m in [("frontier  ", b["frontier"]),
                         ("specialist", b["specialist"])]:
            print(f"  {label}: N={m['n']:3d}  "
                  f"r={m['r']:+.3f}  R²={m['r2_platt']:+.3f}  "
                  f"MAE={m['mae']:.3f}  PI80={m['pi80_cov']:.2f}  "
                  f"width={m['pi80_width']:.3f}")
        print(f"  Δ:           "
              f"r={fmt(b['delta'].get('r'))}  "
              f"R²={fmt(b['delta'].get('r2_platt'))}  "
              f"MAE={fmt(b['delta'].get('mae'))}  "
              f"PI80={fmt(b['delta'].get('pi80_cov'))}")
        print(f"  → routing: {b['routing'].upper()}")

    if blocks:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_report(blocks, args.output_dir / "specialist_vs_frontier.md")
        (args.output_dir / "specialist_vs_frontier.json").write_text(
            json.dumps(blocks, indent=2))
        print(f"\nWrote {args.output_dir}/specialist_vs_frontier.md")


if __name__ == "__main__":
    main()
