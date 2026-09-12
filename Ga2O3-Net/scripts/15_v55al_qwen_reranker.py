"""Phase 55 V55-AL — Qwen Ensemble Re-Ranker for V54-E1 LCMD Shortlist.

Combines LCMD diversity-aware score with Qwen-based chemistry plausibility
scoring to re-rank the 10-recipe wet-lab shortlist. Replaces external hosted-API services
/ GPT-4o / Gemini ensemble with local Qwen2.5-7B (+ optional 2nd run with
different temperature seed for ensemble diversity).

For each recipe, Qwen scores 4 axes (0-1):
  a) likely PDR improvement vs Mg baseline
  b) likely V_O controllability
  c) novelty given N=14 training data
  d) experimental feasibility

Final score = α · LCMD_score + (1-α) · Qwen_aggregate_score (mean of 4 axes).
Tune α on V54-A2 holdout if needed.

Output: results/phase55v55al/ranked_recipes.csv with both LCMD + Qwen breakdowns
and disagreement table.

Gate: post-wet-lab Spearman ρ(LLM rank, measured PDR) ≥ 0.5 (deferred to Phase C).
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_al")


SYSTEM_PROMPT = """You are a β-Ga₂O₃ sputter-deposited photodetector expert.
You score candidate sputter recipes on 4 axes, returning ONLY a valid JSON
object with keys (pdr_score, vo_score, novelty_score, feasibility_score),
each a float in [0, 1]. NO prose, NO markdown."""


SCORING_PROMPT = """Candidate sputter recipe for β-Ga2O3 thin-film
photodetector: {recipe_json}

Given the V54-C1 closed-form V_O predictor:
  V_O_offset ≈ ((ox_state - lone_pair) / exp(0.10·hardness + r_mm) - chi_mm) / 4.66
              - walsh_D + 0.088

and the N=14 sputter V_O training set (which covers Mg, Sn, Si, Zn, Fe, Cu,
Ge, Bi, Sb, Ti, Ta, Er, Eu, Al — NOT Hf, Sc, Y, RE that are mentioned as
unseen in 2023-2026 sputter literature), score this recipe on 4 axes:

(1) `pdr_score` (0-1): How likely is this recipe to give higher PDR (Iph/Idark)
    than a baseline Mg-doped sputter sample (PDR ≈ 10³)? Consider dopant
    electronic effect, anneal optimization, atmosphere.

(2) `vo_score` (0-1): How controllable is V_O concentration likely to be in
    this recipe? Consider Ar:O₂ ratio, anneal atmosphere, dopant compensation.
    Higher score = more reproducible / easier to land in [10¹⁷, 10¹⁸] cm⁻³.

(3) `novelty_score` (0-1): How novel is this recipe vs N=14 training data?
    Unseen dopants (Hf, Sc, Y, RE, Bi, Cu) → score ≥ 0.7. Process variations
    (high p_O2, plasma anneal) → +0.1 to +0.2. Same-as-training → 0.0.

(4) `feasibility_score` (0-1): How feasible is this recipe in a standard RF
    magnetron sputter lab? Score ≥ 0.7 for standard targets / conditions.
    Score < 0.4 for exotic targets (mosaic, multi-target codope), unusual
    pressures (< 0.5 Pa), or unstable substrate combinations.

OUTPUT ONLY: {{"pdr_score": x.x, "vo_score": x.x, "novelty_score": x.x, "feasibility_score": x.x}}"""


# V54-E1 LCMD-generated 10-recipe shortlist (extracted from
# results/phase54v54e1_al/lcmd_recommendations.csv + V54 roadmap §AL Batch)
DEFAULT_RECIPES = [
    {"goal": "OOD donor",          "dopant": "Hf", "c_at_pct": 1.5, "T_sub_C": 25,  "p_O2_pct": 10, "anneal_atm": "Ar",   "anneal_T_C": 800, "substrate": "c-sap",  "lcmd_score": 0.95},
    {"goal": "OOD lone-pair",      "dopant": "Bi", "c_at_pct": 2.0, "T_sub_C": 300, "p_O2_pct": 20, "anneal_atm": "Ar",   "anneal_T_C": 700, "substrate": "c-sap",  "lcmd_score": 0.97},
    {"goal": "OOD acceptor",       "dopant": "Cu", "c_at_pct": 0.8, "T_sub_C": 25,  "p_O2_pct": 0,  "anneal_atm": "N2",   "anneal_T_C": 800, "substrate": "c-sap",  "lcmd_score": 0.92},
    {"goal": "high p_O2",          "dopant": "Sn", "c_at_pct": 1.0, "T_sub_C": 400, "p_O2_pct": 50, "anneal_atm": "O2",   "anneal_T_C": 800, "substrate": "c-sap",  "lcmd_score": 0.85},
    {"goal": "O2 plasma post-anneal","dopant": "Si","c_at_pct": 2.0, "T_sub_C": 25,  "p_O2_pct": 10, "anneal_atm": "Ar+O2-plasma", "anneal_T_C": 800, "substrate": "c-sap", "lcmd_score": 0.83},
    {"goal": "OOD group III",      "dopant": "Sc", "c_at_pct": 1.5, "T_sub_C": 500, "p_O2_pct": 5,  "anneal_atm": "Ar",   "anneal_T_C": 800, "substrate": "r-sap",  "lcmd_score": 0.94},
    {"goal": "low-conc plasma O2", "dopant": "Sn", "c_at_pct": 0.14,"T_sub_C": 25,  "p_O2_pct": 100,"anneal_atm": "N2",   "anneal_T_C": 700, "substrate": "r-sap",  "lcmd_score": 0.80},
    {"goal": "alt acceptor",       "dopant": "Mg", "c_at_pct": 6.0, "T_sub_C": 25,  "p_O2_pct": 10, "anneal_atm": "Ar",   "anneal_T_C": 800, "substrate": "Si",     "lcmd_score": 0.75},
    {"goal": "super-donor",        "dopant": "W",  "c_at_pct": 9.4, "T_sub_C": 25,  "p_O2_pct": 50, "anneal_atm": "none", "anneal_T_C": 0,   "substrate": "c-sap",  "lcmd_score": 0.78},
    {"goal": "isovalent + plasma", "dopant": "Al", "c_at_pct": 0.7, "T_sub_C": 25,  "p_O2_pct": 100,"anneal_atm": "Ar+O2-plasma", "anneal_T_C": 0,  "substrate": "MgO",   "lcmd_score": 0.72},
]


def _score_with_qwen(chat, recipes: list[dict]) -> list[dict]:
    """Call Qwen for each recipe; return list of {pdr_score, vo_score, novelty_score, feasibility_score}."""
    results = []
    for i, recipe in enumerate(recipes):
        clean = {k: v for k, v in recipe.items() if k != "lcmd_score"}
        prompt = SCORING_PROMPT.format(recipe_json=json.dumps(clean, indent=2))
        data = chat.extract_json(SYSTEM_PROMPT, prompt, max_retries=2)
        if data is None or not isinstance(data, dict):
            logger.warning(f"  Recipe {i + 1} ({clean['dopant']}): Qwen failed; using neutral 0.5 default")
            data = {"pdr_score": 0.5, "vo_score": 0.5, "novelty_score": 0.5,
                    "feasibility_score": 0.5}
        # Clip to [0, 1] and fill missing
        for k in ("pdr_score", "vo_score", "novelty_score", "feasibility_score"):
            try:
                v = float(data.get(k, 0.5))
                data[k] = max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                data[k] = 0.5
        logger.info(f"  Recipe {i + 1} {clean['dopant']:>3s} @ {clean.get('c_at_pct', 0):.1f}%: "
                    f"pdr={data['pdr_score']:.2f} vo={data['vo_score']:.2f} "
                    f"nov={data['novelty_score']:.2f} feas={data['feasibility_score']:.2f}")
        results.append(data)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-AL Qwen ensemble re-ranker on V54-E1 LCMD shortlist")
    parser.add_argument("--qwen-model", default="checkpoints/qwen2.5-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="α weight: final_score = α·LCMD + (1-α)·Qwen_aggregate")
    parser.add_argument("--ensemble-seeds", type=int, default=2,
                        help="Number of Qwen runs (different temperature seeds) for ensemble")
    parser.add_argument("--output", type=Path,
                        default=PROJ / "results" / "phase55v55al" / "ranked_recipes.csv")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    recipes = DEFAULT_RECIPES
    logger.info(f"Re-ranking {len(recipes)} V54-E1 shortlist recipes with Qwen ensemble")

    from src.llm import QwenChat

    # Run Qwen `ensemble_seeds` times with different temperatures
    ensemble = []
    for seed_idx in range(args.ensemble_seeds):
        temp = 0.0 if seed_idx == 0 else 0.3
        logger.info(f"=== Qwen pass {seed_idx + 1}/{args.ensemble_seeds} (temperature={temp}) ===")
        chat = QwenChat(model_path=args.qwen_model, device=args.device,
                        dtype="bfloat16", temperature=temp, seed=42 + seed_idx)
        scores = _score_with_qwen(chat, recipes)
        ensemble.append(scores)

    # Aggregate across seeds (mean)
    avg_scores = []
    for i in range(len(recipes)):
        agg = {}
        for k in ("pdr_score", "vo_score", "novelty_score", "feasibility_score"):
            vals = [pass_[i][k] for pass_ in ensemble]
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        agg["qwen_aggregate"] = float(np.mean([agg[k] for k in
                                               ("pdr_score", "vo_score", "novelty_score", "feasibility_score")]))
        avg_scores.append(agg)

    # Combine with LCMD
    rows = []
    for recipe, qwen in zip(recipes, avg_scores):
        row = dict(recipe)
        row.update(qwen)
        row["final_score"] = args.alpha * recipe["lcmd_score"] + (1 - args.alpha) * qwen["qwen_aggregate"]
        row["disagreement"] = abs(recipe["lcmd_score"] - qwen["qwen_aggregate"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    logger.info(f"\nWrote ranked recipes to {args.output}")

    # Print summary
    print("\n=== V55-AL Ranked Recipes (α={:.2f}) ===".format(args.alpha))
    cols_to_show = ["dopant", "c_at_pct", "T_sub_C", "p_O2_pct", "anneal_atm",
                    "anneal_T_C", "lcmd_score", "qwen_aggregate", "final_score",
                    "disagreement"]
    print(df[cols_to_show].to_string(index=False))

    # Disagreement table
    print("\n=== Top-5 LCMD-Qwen Disagreement ===")
    dis = df.sort_values("disagreement", ascending=False).head(5)
    print(dis[["dopant", "c_at_pct", "lcmd_score", "qwen_aggregate", "disagreement"]].to_string(index=False))

    # Save metadata
    meta_path = args.output.parent / "v55al_metadata.json"
    meta = {
        "alpha": args.alpha,
        "ensemble_seeds": args.ensemble_seeds,
        "qwen_model": args.qwen_model,
        "n_recipes": len(recipes),
        "top_3": df.head(3)[["dopant", "c_at_pct", "final_score"]].to_dict(orient="records"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
