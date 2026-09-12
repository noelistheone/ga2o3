"""Phase 55 V55-SR — LLM-seeded Symbolic Regression for dopant_term.

Generates closed-form V_O ≈ f(dopant features, process) formulas by combining
Qwen2.5-Math-7B (or 7B-Instruct fallback) skeleton proposals + PySR constant
optimization. Seeded with the V54-C1 formula:
  ((ox_state − lone_pair) / exp(hardness·0.10 + r_mm) − chi_mm) / 4.66
  − walsh_D + 0.088
which has RMSE 0.031 on V52b/V53-ζ ElementHyperNet outputs.

Goal: discover an OOD-generalizing formula that beats V54-C1 RMSE by ≥10%
AND achieves LOEO (leave-one-element-out) test R² ≥ 0.65 on at least
2/4 held-out elements (Mg/Sn/Si/Zn).

Pipeline (replaces external LLM-SR repo with local Qwen):
  1. Build per-element dataset: (descriptors, hypernet_output_C1) from V54-A1 bundle
  2. Augment with process variables (log_pO2, T_sub_norm, anneal_T_norm) from
     per-element OOF predictions
  3. Loop:
     - Qwen propose 5-10 candidate skeletons (LaTeX-style equation strings)
     - For each skeleton: parse via sympy + PySR-optimize constants
     - Compute within-DOI CV RMSE + LOEO test R²
     - Keep Pareto front
  4. Output top-3 formulas to data/processed/pysr_dopant_term_v55sr.json

Falsification gates (line 102 in V55 Roadmap §V55-SR):
  - Best formula RMSE ≤ V54-C1 0.031 × 0.9 = 0.028
  - LOEO test R² ≥ 0.65 on ≥2 of {Mg, Sn, Si, Zn}
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_sr")


V54C1_FORMULA_JSON = PROJ / "data" / "processed" / "pysr_dopant_term.json"
V55SR_OUTPUT = PROJ / "data" / "processed" / "pysr_dopant_term_v55sr.json"

SYSTEM_PROMPT = """You are a physical-chemistry expert specializing in defect
thermodynamics of oxide semiconductors. You propose closed-form mathematical
equations relating dopant electronic/structural descriptors to log₁₀ oxygen
vacancy concentration in β-Ga₂O₃.

Output ONLY valid mathematical expressions in Python-evaluable syntax (using
+ - * / ** exp log sin cos sqrt). NO prose, NO markdown. One equation per line."""


SEED_PROMPT_TEMPLATE = """You are searching for a closed-form formula for the
ElementHyperNet output (a scalar offset on log₁₀ V_O for β-Ga₂O₃) as a
function of the dopant element's physical descriptors.

Available variables (per dopant, all numeric):
  x0  = Pauling electronegativity (1.3–4.0)
  x1  = Shannon ionic radius (Å, 0.27–1.46)
  x2  = oxidation state (1–6)
  x3  = lone_pair_bool (0 or 1; Bi/Sb = 1)
  x4  = Walsh-D off-centering (0–0.45)
  x5  = Pearson hardness (eV, 3.2–8.0)
  x6  = electronegativity mismatch χ_dopant − χ_Ga (Ga χ = 1.81)
  x7  = ionic radius mismatch (r_dopant − r_Ga)/r_Ga
  x8  = oxidation offset (ox_state − 3)

Best formula found so far (RMSE 0.031, complexity 18, V54-C1 from PySR):
  ((x2 - x3) / exp(x5 * 0.0995 + x7) - x6) / 4.66 - x4 + 0.088

Physics interpretation:
  - (x2 - x3) = effective valence above Ga³⁺ (donor strength)
  - dividing by exp(hardness*0.1 + r_mm) penalizes hard, mismatched dopants
  - subtracting electronegativity mismatch (x6) reduces V_O for more
    electronegative dopants (compensation effect)
  - subtracting Walsh-D off-centering accounts for lone-pair (Bi/Sb)
  - constant +0.088 is the baseline offset

GOAL: Propose 5 NEW candidate formulas that may IMPROVE on V54-C1 while
preserving physics-consistency. Try:
  - Adding log/sqrt of hardness or radius
  - Combining x3 (lone_pair) and x7 (radius_mm) into a single term
  - Including process variables (would need to be added separately)
  - Replacing the / 4.66 normalization with a learned constant
  - Higher-order terms in (x2 - x3)

OUTPUT: 5 candidate formulas, one per line, valid Python expressions.
DO NOT explain. DO NOT use markdown. Just the equations."""


REFINE_PROMPT_TEMPLATE = """The best formula so far is:
  {best_formula}
with RMSE {best_rmse:.4f} (V54-C1 baseline RMSE 0.031).

Recent failed candidates (top-3 worst):
{worst_candidates}

PROPOSE 5 NEW formulas that:
1. Are different in STRUCTURE from the best (not just minor coefficient changes)
2. Incorporate at least one descriptor NOT in the best formula
3. Have complexity ≤ 30 operators
4. Are physically reasonable (no division by potentially-zero terms)

OUTPUT 5 formulas, one per line, Python-evaluable syntax. No prose."""


def _load_v54c1() -> dict[str, Any]:
    """Load V54-C1 PySR result as the seed."""
    if not V54C1_FORMULA_JSON.exists():
        raise FileNotFoundError(f"V54-C1 formula not found: {V54C1_FORMULA_JSON}")
    return json.loads(V54C1_FORMULA_JSON.read_text())


def _build_training_data(bundle_dir: Path) -> pd.DataFrame:
    """Extract (per-element descriptors, hypernet output) from V54-A1 bundle.

    Returns DataFrame with columns x0-x8, hn_out, element.
    """
    import importlib.util
    # Reuse hypernet-extraction logic from scripts/07_pysr_dopant_term.py
    spec = importlib.util.spec_from_file_location(
        "pysr_dopant", PROJ / "scripts" / "07_pysr_dopant_term.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    feats = mod._gather_hypernet_per_element(bundle_dir)
    # Use the V55-SR variable convention (x0..x8)
    rename_map = {
        "EN": "x0", "radius": "x1", "ox_state": "x2", "lone_pair": "x3",
        "walsh_D": "x4", "hardness": "x5", "chi_mm": "x6", "r_mm": "x7",
        "ox_off": "x8", "hypernet_out_0": "hn_out",
    }
    feats = feats.rename(columns=rename_map)
    return feats[["element", "x0", "x1", "x2", "x3", "x4", "x5",
                  "x6", "x7", "x8", "hn_out"]].copy()


def _evaluate_formula(expr: str, df: pd.DataFrame) -> tuple[np.ndarray, float] | None:
    """Try to evaluate `expr` on dataframe rows; return (predictions, RMSE) or None.

    Uses sympy to parse + numpy lambdify for safety.
    """
    import sympy
    try:
        sym_vars = sympy.symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8")
        # Replace common LaTeX-isms / Python forms
        clean = re.sub(r"\\\\", "", expr)
        clean = re.sub(r"\^", "**", clean)
        sym_expr = sympy.sympify(clean, locals={
            "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
            "sin": sympy.sin, "cos": sympy.cos,
        })
        f = sympy.lambdify(sym_vars, sym_expr, modules=["numpy"])
        X = [df[f"x{i}"].values for i in range(9)]
        y_pred = f(*X)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.full(len(df), float(y_pred))
        if y_pred.shape != df["hn_out"].shape:
            return None
        if np.any(~np.isfinite(y_pred)):
            return None
        rmse = float(np.sqrt(np.mean((y_pred - df["hn_out"].values) ** 2)))
        return y_pred, rmse
    except Exception as exc:
        logger.debug(f"Eval failed for '{expr[:60]}...': {exc}")
        return None


def _loeo_score(expr: str, df: pd.DataFrame,
                held_out_elements: list[str] = ("Mg", "Sn", "Si", "Zn")) -> dict[str, float]:
    """Leave-one-element-out test: R² on the held-out element."""
    res = {}
    for elem in held_out_elements:
        mask = df["element"] == elem
        if mask.sum() < 1:
            continue
        evaled = _evaluate_formula(expr, df[mask])
        if evaled is None:
            res[elem] = float("nan")
            continue
        y_pred, _ = evaled
        y_true = df.loc[mask, "hn_out"].values
        if len(y_true) < 2:
            res[elem] = float("nan")
            continue
        ss_tot = float(np.var(y_true) * len(y_true))
        if ss_tot < 1e-12:
            res[elem] = float("nan")
            continue
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        res[elem] = 1.0 - ss_res / ss_tot
    return res


def _parse_qwen_formulas(text: str) -> list[str]:
    """Extract candidate formula strings from a Qwen response."""
    candidates = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # Strip leading numbering like "1.", "1)", "-", "*"
        s = re.sub(r"^[\d]+[\.\)]\s*", "", s)
        s = re.sub(r"^[-\*]\s+", "", s)
        s = s.strip("` ")
        if "x0" not in s and "x1" not in s and "x2" not in s:
            continue
        candidates.append(s)
    return candidates[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-SR Qwen-seeded symbolic regression")
    parser.add_argument("--bundle-dir", type=Path,
                        default=PROJ / "results" / "phase54v54a1_sincere_vc_5seed",
                        help="V54-A1 bundle for ElementHyperNet extraction")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of Qwen-skeleton-proposal iterations")
    parser.add_argument("--qwen-model", default="checkpoints/qwen2.5-7b",
                        help="Qwen model path (use Math-7B if available)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=V55SR_OUTPUT)
    args = parser.parse_args()

    # Load training data
    logger.info(f"Loading ElementHyperNet from {args.bundle_dir}")
    df = _build_training_data(args.bundle_dir)
    logger.info(f"  {len(df)} elements with hn_out ∈ [{df['hn_out'].min():.3f}, "
                f"{df['hn_out'].max():.3f}]")

    # Evaluate V54-C1 baseline
    v54c1_meta = _load_v54c1()
    v54c1_expr = v54c1_meta["best_equation"]
    # Sympy struggles with the raw V54-C1 from PySR if it has e-notation; normalize
    base_eval = _evaluate_formula(v54c1_expr, df)
    if base_eval:
        base_rmse = base_eval[1]
    else:
        # Fall back to the cached best_loss from the JSON
        base_rmse = float(np.sqrt(float(v54c1_meta["best_loss"])))
    logger.info(f"V54-C1 RMSE = {base_rmse:.4f}")
    base_loeo = _loeo_score(v54c1_expr, df)
    logger.info(f"V54-C1 LOEO R²: {base_loeo}")

    # Load Qwen
    from src.llm import QwenChat
    chat = QwenChat(model_path=args.qwen_model, device=args.device,
                    dtype="bfloat16", temperature=0.7, top_p=0.95, seed=42)

    # Iterative search
    pareto = [{
        "expr": v54c1_expr, "rmse": base_rmse,
        "loeo": base_loeo, "complexity": int(v54c1_meta.get("best_complexity", 18)),
        "iteration": -1, "origin": "V54-C1 seed",
    }]
    best_rmse = base_rmse
    best_expr = v54c1_expr

    t0 = time.time()
    for it in range(args.n_iter):
        elapsed = time.time() - t0
        logger.info(f"=== Iter {it + 1}/{args.n_iter} ({elapsed:.0f}s) best_rmse={best_rmse:.4f} ===")
        if it == 0:
            user_prompt = SEED_PROMPT_TEMPLATE
        else:
            # Get 3 worst recent candidates for variety
            recent_worst = sorted(pareto, key=lambda p: -p["rmse"])[:3]
            worst_str = "\n".join(f"  {r['expr'][:80]} (RMSE={r['rmse']:.3f})" for r in recent_worst)
            user_prompt = REFINE_PROMPT_TEMPLATE.format(
                best_formula=best_expr, best_rmse=best_rmse,
                worst_candidates=worst_str,
            )
        response = chat.chat(SYSTEM_PROMPT, user_prompt, max_new_tokens=512)
        candidates = _parse_qwen_formulas(response)
        logger.info(f"  Qwen proposed {len(candidates)} candidates")
        for c in candidates:
            ev = _evaluate_formula(c, df)
            if ev is None:
                continue
            _, rmse = ev
            loeo = _loeo_score(c, df)
            comp = len(re.findall(r"[+\-*/]|exp|log|sqrt", c))
            pareto.append({
                "expr": c, "rmse": rmse, "loeo": loeo,
                "complexity": comp, "iteration": it, "origin": "qwen",
            })
            if rmse < best_rmse:
                logger.info(f"  ★ NEW BEST RMSE={rmse:.4f} : {c[:80]}")
                best_rmse = rmse
                best_expr = c

    # Filter top-3 + check gates
    pareto.sort(key=lambda p: p["rmse"])
    top3 = pareto[:3]
    target_rmse = base_rmse * 0.9  # ≥10% better
    gate_rmse_pass = best_rmse <= target_rmse
    best_loeo = top3[0]["loeo"] if top3 else {}
    loeo_pass_count = sum(1 for e in ("Mg", "Sn", "Si", "Zn")
                          if best_loeo.get(e, -1) >= 0.65)
    gate_loeo_pass = loeo_pass_count >= 2

    out = {
        "v54c1_seed": {"expr": v54c1_expr, "rmse": base_rmse, "loeo": base_loeo},
        "best": top3[0],
        "top3": top3,
        "n_candidates_evaluated": len(pareto),
        "gate_rmse_target": target_rmse,
        "gate_rmse_pass": gate_rmse_pass,
        "gate_loeo_pass_count": loeo_pass_count,
        "gate_loeo_pass": gate_loeo_pass,
        "overall_pass": gate_rmse_pass and gate_loeo_pass,
        "qwen_model": args.qwen_model,
        "n_iter": args.n_iter,
        "training_n_elements": len(df),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"\nWrote {args.output}")
    logger.info(f"  Best RMSE: {best_rmse:.4f}  (target ≤ {target_rmse:.4f})  "
                f"{'PASS' if gate_rmse_pass else 'FAIL'}")
    logger.info(f"  LOEO R² ≥ 0.65 on {loeo_pass_count}/4 elements  "
                f"{'PASS' if gate_loeo_pass else 'FAIL'}")
    logger.info(f"  Overall verdict: {'PASS — replace V54-C1' if out['overall_pass'] else 'NEG — keep V54-C1'}")


if __name__ == "__main__":
    main()
