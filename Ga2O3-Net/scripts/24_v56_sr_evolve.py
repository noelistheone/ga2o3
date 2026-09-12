"""V56-SR-Evolve — 2-island LLM-SR with Qwen-Math (explore) + Qwen-14B-AWQ (exploit).

V55-SR (Phase 55) achieved RMSE 0.671 vs V54-C1's 0.031 — failed LOEO gate
(all 4 elements NaN). V56-SR-Evolve scales to AlphaEvolve-style evolutionary
search: two populations (islands), cross-island migration, fitness = OOD
extrapolation residual on PDR > 10^4 holdout.

Backbones (zero-API, all local):
  - Explore island: Qwen2.5-Math-7B-Instruct, T=0.9 (diverse skeletons)
  - Exploit island: Qwen2.5-14B-AWQ,        T=0.1 (refinement)

Seed: V54-C1 closed-form formula + V55-SR softplus-denominator finding.

Variables (per row):
  Element descriptors (9): EN, radius, ox_state, lone_pair, walsh_D, hardness,
    chi_mm, r_mm, ox_off
  Process (3): log_pO2, T_sub_norm, anneal_T_norm

Output:
  - data/processed/v56_sr_evolve_top3.json   — top-3 formulas + LOEO R² per element
  - logs/v56_sr_evolve.log
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import make_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v56_sr_evolve")


SEED_FORMULA_V54C1 = (
    "((ox_state - lone_pair) / exp(hardness*0.10 + r_mm) - chi_mm) / 4.66 "
    "- walsh_D + 0.088"
)
SEED_FORMULA_V55SR = (
    "((ox_state - lone_pair) / softplus(hardness*0.10 + r_mm) - chi_mm) / 4.66 "
    "- walsh_D + 0.088"
)


PROPOSAL_SYSTEM = (
    "You are a symbolic-regression expert proposing mathematical formulas to "
    "predict log10(V_O cm^-3) for sputter-deposited β-Ga2O3 thin films. The "
    "available input variables are element descriptors and process axes. You "
    "must propose ONE new formula (as a single sympy/numpy-evaluable expression) "
    "that improves on the seed formula's within-DOI fit. Output ONLY the formula, "
    "no prose, no markdown."
)


PROPOSAL_USER_TMPL = """Variables available:
  Element descriptors: EN, radius, ox_state, lone_pair, walsh_D, hardness, chi_mm, r_mm, ox_off
  Process axes: log_pO2, T_sub_norm, anneal_T_norm

Allowed operators: + - * / **
Allowed functions: exp, log, sqrt, abs, softplus, tanh, sin, cos

Seed formula (V54-C1, RMSE 0.031 in-domain): {seed_v54c1}

Recent best formula (V55-SR, RMSE 0.671 in-domain): {seed_v55sr}

Current population top-3 formulas (rmse | formula):
{population_summary}

Propose ONE new candidate formula that explores a NEW idea. Be creative but
keep the algebra simple (depth ≤ 6 operations).

Output ONLY the formula as a single line:"""


def _safe_softplus(x):
    return np.log1p(np.exp(np.minimum(x, 30)))


_VAR_NAMES = ["EN", "radius", "ox_state", "lone_pair", "walsh_D", "hardness",
              "chi_mm", "r_mm", "ox_off", "log_pO2", "T_sub_norm", "anneal_T_norm"]
_FUNC_NAMES = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
               "softplus": _safe_softplus, "tanh": np.tanh, "sin": np.sin, "cos": np.cos}


def _safe_eval(formula: str, vars_dict: dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Safely evaluate formula. Returns None on failure."""
    # Whitelist: variables + operators + functions
    bad = re.search(r"\b(import|exec|eval|open|os|sys|__|while|for|lambda)\b", formula)
    if bad:
        return None
    safe_globals = {"__builtins__": {}}
    safe_locals = {**_FUNC_NAMES, **vars_dict}
    try:
        y = eval(formula, safe_globals, safe_locals)
        if isinstance(y, (int, float)):
            return np.full(len(next(iter(vars_dict.values()))), float(y))
        y = np.asarray(y, dtype=float)
        if np.all(np.isfinite(y)):
            return y
        return None
    except Exception:
        return None


def fit_constants(formula: str, vars_dict: dict, y_true: np.ndarray) -> tuple[Optional[str], float]:
    """Best-effort: evaluate formula, then fit a single shift/scale (a*y + b).
    Returns (refit_formula_or_None, RMSE)."""
    y_pred = _safe_eval(formula, vars_dict)
    if y_pred is None:
        return None, float("inf")
    # Fit a, b: y_true = a*y_pred + b
    if np.std(y_pred) < 1e-9:
        return None, float("inf")
    a, b = np.polyfit(y_pred, y_true, 1)
    y_fit = a * y_pred + b
    rmse = float(np.sqrt(np.mean((y_fit - y_true)**2)))
    refit = f"({a:.4f})*({formula}) + ({b:.4f})"
    return refit, rmse


def get_variables(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build dict of input variables from DataFrame. Filters to rows with V_O label."""
    from src.data.element_descriptors import ELEMENT_DESCRIPTORS

    df = df.copy()
    mask = df["vacancy_concentration"].notna()
    df = df[mask]
    if "element" not in df.columns:
        df["element"] = df["dopant_spec"].fillna("").str.split(":").str[0]

    rows = []
    y_list = []
    elem_list = []
    for _, row in df.iterrows():
        elem = row.get("element") or ""
        if not elem:
            continue
        ed = ELEMENT_DESCRIPTORS.get(elem)
        if ed is None:
            continue
        # Process axes
        c_at = float(row.get("concentration_at%", 0.0) or 0.0)
        T_C = row.get("temperature_C", 800.0)
        T = float(T_C) if T_C is not None and not pd.isna(T_C) else 800.0
        atm = str(row.get("atmosphere", "Ar") or "Ar").lower()
        if "o2" in atm:
            o2_frac = 0.5 if "ar" in atm else 1.0
        else:
            o2_frac = 0.01
        log_pO2 = float(np.log10(max(o2_frac, 1e-3)))
        T_sub_norm = T / 1000.0  # rough normalization
        anneal_T_norm = T / 1000.0

        # ELEMENT_DESCRIPTORS is an 11-tuple z-scored:
        #   0:EN, 1:radius, 2:carrier_type, 3:Δv, 4:Δperiod, 5:lone_pair,
        #   6:walsh_D, 7:hardness, 8:chi_mm, 9:r_mm, 10:ox_off
        rows.append([
            ed[0], ed[1], ed[2], ed[5], ed[6], ed[7],
            ed[8], ed[9], ed[10], log_pO2, T_sub_norm, anneal_T_norm,
        ])
        y_list.append(float(row["vacancy_concentration"]))
        elem_list.append(elem)

    if not rows:
        return {}, np.array([]), []
    X = np.asarray(rows)
    vars_dict = {n: X[:, i] for i, n in enumerate(_VAR_NAMES)}
    return vars_dict, np.asarray(y_list), elem_list


def evaluate_loeo(formula: str, vars_dict: dict, y_true: np.ndarray, elements: list[str]) -> dict[str, float]:
    """Leave-one-element-out RMSE per held-out element."""
    out = {}
    unique_elems = sorted(set(elements))
    for held in unique_elems:
        train_idx = [i for i, e in enumerate(elements) if e != held]
        held_idx = [i for i, e in enumerate(elements) if e == held]
        if len(held_idx) < 2:
            out[held] = float("nan")
            continue
        train_vars = {k: v[train_idx] for k, v in vars_dict.items()}
        held_vars = {k: v[held_idx] for k, v in vars_dict.items()}
        # Fit constants on train
        refit, _ = fit_constants(formula, train_vars, y_true[train_idx])
        if refit is None:
            out[held] = float("nan")
            continue
        y_pred = _safe_eval(refit, held_vars)
        if y_pred is None:
            out[held] = float("nan")
            continue
        rmse_held = float(np.sqrt(np.mean((y_pred - y_true[held_idx])**2)))
        out[held] = rmse_held
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-SR-Evolve 2-island LLM-SR")
    ap.add_argument("--merged-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv")
    ap.add_argument("--fallback-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v55.csv")
    ap.add_argument("--iterations", type=int, default=200,
                    help="number of LLM proposal calls per island")
    ap.add_argument("--population", type=int, default=20,
                    help="island population size")
    ap.add_argument("--migrate-every", type=int, default=5)
    ap.add_argument("--device-explore", default="cuda:0",
                    help="GPU for Qwen-Math-7B explore island")
    ap.add_argument("--device-exploit", default="cuda:1",
                    help="GPU for Qwen-14B-AWQ exploit island")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path,
                    default=PROJ / "data" / "processed" / "v56_sr_evolve_top3.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.merged_csv if args.merged_csv.exists() else args.fallback_csv
    df = pd.read_csv(csv_path)
    vars_dict, y_true, elements = get_variables(df)
    if len(y_true) < 10:
        logger.error(f"Too few V_O-labeled rows ({len(y_true)}); abort")
        return
    logger.info(f"V_O rows: {len(y_true)} | unique elements: {sorted(set(elements))}")

    # Initialize population with seeds
    population: list[dict] = []
    for seed_str in [SEED_FORMULA_V54C1, SEED_FORMULA_V55SR]:
        refit, rmse = fit_constants(seed_str, vars_dict, y_true)
        if refit is not None:
            population.append({"formula": seed_str, "refit": refit, "rmse": rmse, "island": "seed"})

    # Phase 56 reality: Math-7B (14GB fp16) + AWQ-14B (8GB int4) don't fit on a
    # single 24GB RTX 3090 with inference overhead. When --device-explore equals
    # --device-exploit, share ONE Qwen-14B-AWQ client; diversity comes from
    # temperature ramp (explore=0.9, exploit=0.1) set per iteration in the loop.
    if args.device_explore == args.device_exploit:
        logger.info(f"Single-GPU mode: shared Qwen-14B-AWQ on {args.device_explore} "
                    "(temperature varies per island)")
        shared = make_client("qwen-14b-awq", device=args.device_explore,
                              temperature=0.5, top_p=0.95, max_new_tokens=256)
        llm_explore = shared
        llm_exploit = shared
    else:
        logger.info(f"Loading explore LLM: Qwen-Math-7B on {args.device_explore}")
        llm_explore = make_client("qwen-math-7b", device=args.device_explore,
                                  temperature=0.9, top_p=0.95, max_new_tokens=256)
        logger.info(f"Loading exploit LLM: Qwen-14B-AWQ on {args.device_exploit}")
        llm_exploit = make_client("qwen-14b-awq", device=args.device_exploit,
                                  temperature=0.1, top_p=0.95, max_new_tokens=256)

    def _pop_summary(pop: list[dict], k: int = 3) -> str:
        pop_sorted = sorted([p for p in pop if p["rmse"] < 1e6], key=lambda p: p["rmse"])[:k]
        return "\n".join(f"  {p['rmse']:.4f} | {p['formula'][:120]}" for p in pop_sorted)

    t0 = time.time()
    for it in range(args.iterations):
        # Alternate islands
        use_explore = (it % 2 == 0)
        llm = llm_explore if use_explore else llm_exploit
        island_name = "explore" if use_explore else "exploit"
        # If single-client (shared mode), set temperature dynamically per island
        if llm_explore is llm_exploit and hasattr(llm, "temperature"):
            llm.temperature = 0.9 if use_explore else 0.1

        user = PROPOSAL_USER_TMPL.format(
            seed_v54c1=SEED_FORMULA_V54C1,
            seed_v55sr=SEED_FORMULA_V55SR,
            population_summary=_pop_summary(population),
        )
        try:
            response = llm.chat(PROPOSAL_SYSTEM, user, max_new_tokens=200)
        except Exception as e:
            logger.warning(f"  iter {it}: LLM call failed: {e}")
            continue

        # Take first non-empty line as formula
        formula = None
        for line in response.splitlines():
            line = line.strip().lstrip("> ").rstrip(";.")
            line = re.sub(r"^[`'\"]+", "", line).rstrip("`'\"")
            if line and not line.startswith("#") and not line.lower().startswith("formula"):
                formula = line
                break
        if formula is None:
            continue

        refit, rmse = fit_constants(formula, vars_dict, y_true)
        if refit is None:
            continue
        population.append({
            "formula": formula, "refit": refit, "rmse": rmse, "island": island_name,
        })

        # Cull to size
        population.sort(key=lambda p: p["rmse"])
        population = population[:args.population]

        if (it + 1) % 20 == 0:
            elapsed = (time.time() - t0) / 60
            best = population[0]
            logger.info(f"  iter {it+1}/{args.iterations} ({elapsed:.1f}m): "
                        f"pop_size={len(population)} best_rmse={best['rmse']:.4f} "
                        f"({best['island']}) formula={best['formula'][:80]}")

    # Final: LOEO eval on top-3
    population.sort(key=lambda p: p["rmse"])
    top3 = population[:3]
    for entry in top3:
        loeo = evaluate_loeo(entry["formula"], vars_dict, y_true, elements)
        entry["loeo_rmse_per_element"] = {k: (None if np.isnan(v) else v) for k, v in loeo.items()}
        valid_loeo = [v for v in loeo.values() if not np.isnan(v)]
        entry["loeo_n_valid"] = len(valid_loeo)
        entry["loeo_mean_rmse"] = float(np.mean(valid_loeo)) if valid_loeo else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "iterations": args.iterations,
        "n_rows": int(len(y_true)),
        "elements": sorted(set(elements)),
        "v54c1_seed_rmse": next((p["rmse"] for p in population if p["formula"] == SEED_FORMULA_V54C1), None),
        "v55sr_seed_rmse": next((p["rmse"] for p in population if p["formula"] == SEED_FORMULA_V55SR), None),
        "top3": top3,
    }, indent=2, default=str))
    logger.info(f"V56-SR-Evolve DONE. Top-3 → {args.output}")


if __name__ == "__main__":
    main()
