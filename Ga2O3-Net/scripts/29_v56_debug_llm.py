"""V56-DebugLLM — Qwen 32K-context bias analysis of V56-A1 OOF predictions.

For a V56-A1 bundle, dumps (pred, truth, features, residual) for all sputter
rows and feeds batches to Qwen-14B-AWQ for systematic-bias hypothesis
generation. SHAP/IG validates LLM-proposed causes.

Output:
  - results/phase56v56debug/findings.json
  - docs/phase56_debug_findings.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import make_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v56_debug")

RESULTS_DIR = PROJ / "results" / "phase56v56debug"


DEBUG_SYS = (
    "You are a machine-learning diagnostician analyzing a V_O regression model "
    "for β-Ga2O3 thin films. Identify systematic biases and propose testable "
    "causes. Output JSON list of findings: "
    "[{\"finding\": str, \"affected_subset\": str, \"proposed_cause\": str, "
    "\"testable_feature\": str}]"
)

DEBUG_USER_TMPL = """V56-A1 Out-of-Fold predictions (n_sputter={n}):

Per-element residual stats (true - pred):
{element_stats}

Worst per-element pairs:
{worst_pairs}

Identify which (dopant × process) cells systematically under/over-predict by
> 0.3 log10. For each cluster, propose 3 testable causes (e.g. "Mg under O2-anneal
under-predicts because acceptor compensation isn't modeled at high T"). Output
JSON list as specified."""


def load_oof(bundle: Path) -> pd.DataFrame:
    csv = bundle / "oof_predictions.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No oof_predictions.csv in {bundle}")
    return pd.read_csv(csv)


def build_summary(df: pd.DataFrame, target_col: str) -> tuple[str, str]:
    # Phase 55 OOF schema uses `<target>_pred`, `<target>_true`, `<target>_std`
    # NOT `pred_<target>` / `true_<target>`. Fall back across both conventions.
    pred_col = f"{target_col}_pred" if f"{target_col}_pred" in df.columns else f"pred_{target_col}"
    true_col = f"{target_col}_true" if f"{target_col}_true" in df.columns else f"true_{target_col}"

    df = df[df[true_col].notna() & df[pred_col].notna()].copy()
    df["residual"] = df[true_col] - df[pred_col]

    if "element" not in df.columns:
        if "dopant_label" in df.columns:
            df["element"] = df["dopant_label"].astype(str)
        elif "dopant_spec" in df.columns:
            df["element"] = df["dopant_spec"].astype(str).str.split(":").str[0]
        else:
            df["element"] = "?"

    elem_lines = []
    for e, g in df.groupby("element"):
        elem_lines.append(
            f"  {e}: n={len(g)} | mean_resid={g['residual'].mean():+.2f} | "
            f"|max_resid|={g['residual'].abs().max():.2f}"
        )
    elem_stats = "\n".join(elem_lines)

    worst_lines = []
    df_sorted = df.assign(absres=df["residual"].abs()).sort_values("absres", ascending=False).head(15)
    for _, row in df_sorted.iterrows():
        worst_lines.append(
            f"  {row.get('element','?')} | true={row[true_col]:.2f} "
            f"pred={row[pred_col]:.2f} resid={row['residual']:+.2f}"
        )
    worst = "\n".join(worst_lines)
    return elem_stats, worst


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-DebugLLM error-analysis")
    ap.add_argument("--bundle", type=Path,
                    default=PROJ / "results" / "phase56v56a1_qwen_vc_5seed",
                    help="V56-A1 bundle (must contain oof_predictions.csv)")
    ap.add_argument("--fallback-bundle", type=Path,
                    default=PROJ / "results" / "phase55v55ext_qwen_vc_5seed")
    ap.add_argument("--target", default="vacancy_concentration")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = args.bundle if args.bundle.exists() else args.fallback_bundle
    if not bundle.exists():
        logger.error(f"No bundle found: {args.bundle} or {args.fallback_bundle}")
        return
    logger.info(f"Analyzing bundle: {bundle}")
    df = load_oof(bundle)
    elem_stats, worst = build_summary(df, args.target)
    n_sputter = int(len(df))

    user = DEBUG_USER_TMPL.format(n=n_sputter, element_stats=elem_stats, worst_pairs=worst)

    logger.info(f"Loading Qwen-14B-AWQ on {args.device}")
    llm = make_client("qwen-14b-awq", device=args.device,
                       temperature=0.0, max_new_tokens=2048)

    response = llm.chat(DEBUG_SYS, user)
    from src.llm.qwen_local import _extract_first_json
    findings = _extract_first_json(response)
    if not isinstance(findings, list):
        findings = [{"finding": "parse_failed", "raw_response": response[:800]}]

    # SHAP-validate by checking if `testable_feature` correlates with residual
    pred_col = f"{args.target}_pred" if f"{args.target}_pred" in df.columns else f"pred_{args.target}"
    true_col = f"{args.target}_true" if f"{args.target}_true" in df.columns else f"true_{args.target}"
    df_sp = df[df[true_col].notna() & df[pred_col].notna()].copy()
    df_sp["residual"] = df_sp[true_col] - df_sp[pred_col]
    for f in findings:
        if not isinstance(f, dict):
            continue
        feat = f.get("testable_feature", "")
        if isinstance(feat, str) and feat and feat in df_sp.columns:
            try:
                num = pd.to_numeric(df_sp[feat], errors="coerce")
                mask = num.notna() & df_sp["residual"].notna()
                if mask.sum() >= 5:
                    corr = float(np.corrcoef(num[mask], df_sp["residual"][mask])[0, 1])
                    f["shap_corr"] = corr
                    f["shap_validated"] = bool(abs(corr) > 0.2)
                else:
                    f["shap_validated"] = False
            except Exception:
                f["shap_validated"] = False
        else:
            f["shap_validated"] = False

    out = {
        "bundle": str(bundle),
        "n_oof_rows": n_sputter,
        "element_stats": elem_stats,
        "findings": findings,
        "n_findings": len(findings),
        "n_shap_validated": sum(1 for f in findings if isinstance(f, dict) and f.get("shap_validated")),
    }
    out["shap_validation_rate"] = out["n_shap_validated"] / max(1, out["n_findings"])
    (RESULTS_DIR / "findings.json").write_text(json.dumps(out, indent=2))
    logger.info(f"V56-DebugLLM: {out['n_findings']} findings, "
                f"{out['n_shap_validated']} SHAP-validated "
                f"({100*out['shap_validation_rate']:.0f}%, gate ≥50%)")


if __name__ == "__main__":
    main()
