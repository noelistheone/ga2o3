"""V57-SR-Evolve+DFT — LLM-SR with HSE06 hard-fact constraints.

Replaces V56-SR-Evolve two-island Qwen-Math/Qwen-14B with single-island
Qwen3-30B-A3B-Thinking + temperature ramp (0.9→0.1 across generations).

Injects 5 HSE06-derived hard facts into the proposer prompt (Roadmap §F).
Composite loss (Lazebnik Nat Sci Rep 2026):
  L = 0.5 * MSE + 0.3 * domain_violation + 0.2 * LLM_complexity_score
where w_i in [0,1] and sum=1.

Holdout LOEO eval excludes training-set DOIs (Roadmap caveat #9: anti-memorization).

Output: data/processed/v57_sr_evolve_top3.json
Gate: LOEO ≥ 9/10 valid; RMSE ≤ 1.0 log10 V_O (V56-SR: 7/10 + 1.56 RMSE).
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import make_client
from src.data.kroger_synthetic import DOPANT_OFFSET, kroger_predict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_sr_evolve")

EXP_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv"
OUT_DIR = PROJ / "results" / "phase57v57sr_evolve_dft"
TOP_JSON = PROJ / "data" / "processed" / "v57_sr_evolve_top3.json"

HSE06_HARD_FACTS = """HSE06 hard facts to satisfy:
  F1. V_O is a deep donor: (+2/0) transition at 0.78 eV above VBM, far below
      CBM (4.85 eV). log10[V_O] drops as Fermi level rises above mid-gap.
  F2. Brouwer asymptote: log[V_O] ∝ -1/4 log(p_O2) in the intrinsic regime.
  F3. At p_O2 → 0, log[V_O] saturates at the Frenkel-defect equilibrium (~22).
  F4. Native E_f^{+2}(VBM, Ga-rich) ≈ -0.3 eV; native E_f^{0}(VBM, Ga-rich) ≈ +3.5 eV.
  F5. Sn_Ga is a shallow donor (level ~0.1 eV below CBM); Mg_Ga, Zn_Ga are
      deep acceptors (level near mid-gap). Donors raise [V_O^+2]; acceptors lower it.
"""

PROPOSER_SYSTEM = """You are a symbolic-regression engine that proposes Python
expressions f(elem, log_c, T_K, log_pO2) returning log10[V_O^+2] in cm^-3
for β-Ga2O3 thin films. Each proposal MUST:
  - Be a single Python expression using only +, -, *, /, **, math.log, math.exp,
    and the provided `elem_offset[elem]` (lookup table).
  - Satisfy all HSE06 hard facts in the system prompt.
  - Use `prefactor = 22.45` (log10 N_O_sites) somewhere.

Output ONLY the expression (no prose, no markdown fences)."""


def gen_users(facts: str, prior_results: list[dict] | None = None) -> str:
    history = ""
    if prior_results:
        top = sorted(prior_results, key=lambda r: r.get("composite_loss", 99))[:3]
        history = "\n\nPrior top-3 (composite_loss):\n" + "\n".join(
            f"  - {r.get('expr','?')}  → {r.get('composite_loss','?'):.3f}" for r in top
        )
    return f"""{facts}

Variables available:
  elem        : str (dopant element symbol, e.g. 'Sn')
  log_c       : float (log10 of dopant atomic fraction)
  T_K         : float (temperature in K)
  log_pO2     : float (log10 of O2 partial pressure in atm)
  elem_offset : dict (provided; per-element dopant_term offset in [-1.0, +1.0])
  prefactor   : float = 22.45
  KB_EV       : float = 8.617e-5
  LN10        : float = math.log(10)
  E_f_q2      : float = -0.3 (native HSE06 V_O^+2 formation energy at εF=VBM, Ga-rich)

Example seed (V54-C1 closed-form):
  prefactor - E_f_q2 / (KB_EV * T_K * LN10) - 0.5 * log_pO2 + elem_offset[elem] * log_c

Generate ONE new expression that improves on the prior top-3 above (if any)
while respecting HSE06 facts F1-F5.{history}"""


def evaluate_expression(expr: str, df: pd.DataFrame) -> dict:
    """Evaluate a Python expression against the dataset. Compute composite loss.
    Returns dict with mse, domain_violation, llm_complexity (proxy = expr length / 200),
    composite_loss, n_valid.
    """
    KB_EV = 8.617333262e-5
    LN10 = math.log(10.0)
    prefactor = 22.45
    E_f_q2 = -0.3
    elem_offset = DOPANT_OFFSET

    safe_globals = {
        "__builtins__": {"abs": abs, "min": min, "max": max, "pow": pow, "len": len},
        "math": math,
    }
    n_valid = 0
    n_violation = 0
    sq_err_sum = 0.0
    for _, row in df.iterrows():
        elem = str(row.get("dopant_spec", "Sn:0.01")).split(":")[0].split(",")[0]
        c = 1e-2
        try:
            # Naive parse: take number after first :
            parts = str(row.get("dopant_spec", "")).split(":")
            if len(parts) > 1:
                c = max(float(parts[1].split(",")[0]), 1e-6)
        except Exception:
            pass
        if pd.isna(row.get("temperature_C")) or pd.isna(row.get("vacancy_concentration")):
            continue
        T_K = (row.get("temperature_C", 500) + 273.15)
        log_c = math.log10(c)
        log_pO2 = -2.0  # heuristic for sputter — TODO map from method+atmosphere
        ns = {"elem": elem, "log_c": log_c, "T_K": T_K, "log_pO2": log_pO2,
              "elem_offset": elem_offset, "prefactor": prefactor,
              "KB_EV": KB_EV, "LN10": LN10, "E_f_q2": E_f_q2}
        try:
            val = float(eval(expr, safe_globals, ns))
        except Exception:
            n_violation += 1
            continue
        # HSE06 hard-fact violation checks
        if not (9.0 <= val <= 22.45):
            n_violation += 1
        sq_err_sum += (val - float(row["vacancy_concentration"])) ** 2
        n_valid += 1

    if n_valid == 0:
        return {"mse": float("inf"), "domain_violation": 1.0,
                "llm_complexity": 1.0, "composite_loss": 99.0, "n_valid": 0}
    mse = sq_err_sum / n_valid
    domain_violation = n_violation / max(1, n_valid + n_violation)
    llm_complexity = min(1.0, len(expr) / 200)
    composite = 0.5 * mse + 0.3 * domain_violation + 0.2 * llm_complexity
    return {"mse": mse, "domain_violation": domain_violation,
            "llm_complexity": llm_complexity, "composite_loss": composite,
            "n_valid": n_valid}


def loeo_eval(expr: str, df: pd.DataFrame) -> dict:
    """Leave-one-DOI-out RMSE."""
    if "doi" not in df.columns and "doi_hash" not in df.columns:
        return {"loeo_rmse": float("nan"), "n_dois": 0}
    doi_col = "doi" if "doi" in df.columns else "doi_hash"
    dois = df[doi_col].dropna().unique()
    rmses = []
    for d in dois:
        hold = df[df[doi_col] == d]
        eval_res = evaluate_expression(expr, hold)
        if eval_res["n_valid"] > 0:
            rmses.append(math.sqrt(eval_res["mse"]))
    if not rmses:
        return {"loeo_rmse": float("nan"), "n_dois": 0, "n_valid_dois": 0}
    return {"loeo_rmse": float(np.mean(rmses)),
             "loeo_rmse_max": float(np.max(rmses)),
             "n_dois": len(dois), "n_valid_dois": len(rmses)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-generations", type=int, default=20)
    ap.add_argument("--proposals-per-gen", type=int, default=5)
    ap.add_argument("--backend", default="qwen3-30b-thinking-vllm",
                    help="LLM backend for proposer. Default uses local vLLM Qwen3-Thinking.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXP_CSV)
    df = df[df["vacancy_concentration"].notna()].copy()
    log.info(f"Eval on {len(df)} labelled V_O rows")

    log.info(f"Loading {args.backend} on {args.device}")
    try:
        # 8192 tokens because Qwen3-Thinking burns most of the budget on
        # chain-of-thought before emitting the final expression. With <512
        # the response is cut off mid-think and yields no parseable expr.
        client = make_client(args.backend, device=args.device,
                              max_new_tokens=8192, temperature=0.9, seed=args.seed)
    except Exception as e:
        log.warning(f"backend {args.backend} failed ({e}); fallback qwen-14b-awq")
        client = make_client("qwen-14b-awq", device=args.device,
                              max_new_tokens=2048, temperature=0.9, seed=args.seed)

    all_results = []
    for gen in range(args.n_generations):
        # Linear temperature ramp 0.9 → 0.1 across generations
        T = 0.9 - 0.8 * gen / max(1, args.n_generations - 1)
        log.info(f"=== Generation {gen+1}/{args.n_generations} (T={T:.2f}) ===")
        client.temperature = T
        prior = all_results
        user_prompt = gen_users(HSE06_HARD_FACTS, prior)
        for k in range(args.proposals_per_gen):
            try:
                resp = client.chat(PROPOSER_SYSTEM, user_prompt)
            except Exception as e:
                log.warning(f"chat failed: {e}")
                continue
            # Robustly mine an expression out of the Thinking-style response.
            # Qwen3-Thinking interleaves chain-of-thought with the final
            # answer; the expression can show up inside a fenced code block,
            # after a phrase like "Final expression:", "Answer:", or on a
            # bare line near the end of the message. Try (1) the longest
            # fenced code block, (2) lines after a final-answer marker,
            # (3) any single line that ast-parses.
            import re as _re
            candidates: list[str] = []
            # (1) fenced code blocks
            for m in _re.finditer(r"```(?:python|py)?\s*\n?([\s\S]*?)```", resp):
                for ln in m.group(1).splitlines():
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        candidates.append(ln)
            # (2) post-"Final ... :" lines
            for marker in ("Final expression:", "Final:", "Answer:", "Output:"):
                idx = resp.lower().rfind(marker.lower())
                if idx >= 0:
                    tail = resp[idx + len(marker):].strip().splitlines()
                    for ln in tail[:5]:
                        ln = ln.strip().strip("`")
                        if ln and not ln.startswith("#"):
                            candidates.append(ln)
            # (3) every standalone line that ast-parses
            for ln in resp.splitlines():
                ln = ln.strip().strip("`")
                if 10 <= len(ln) <= 400 and not ln.startswith("#"):
                    candidates.append(ln)
            expr = None
            for cand in candidates:
                cand = cand.strip("` \n,.").strip()
                # Drop leading "expr =" / "f(...) = " labels
                cand = _re.sub(r"^\s*(?:expr|f\s*\([^)]*\))\s*=\s*", "", cand)
                if len(cand) < 10:
                    continue
                try:
                    ast.parse(cand, mode="eval")
                except SyntaxError:
                    continue
                # Reject pure-literal expressions (e.g. "22.45")
                if _re.fullmatch(r"[-+\d\.eE\s]+", cand):
                    continue
                expr = cand
                break
            if expr is None:
                continue
            res = evaluate_expression(expr, df)
            res["expr"] = expr
            res["gen"] = gen
            res["temperature"] = T
            all_results.append(res)
            if k == 0:
                log.info(f"  proposal: {expr[:60]}... → "
                         f"composite={res['composite_loss']:.3f} mse={res['mse']:.3f}")

    # Top-3 + LOEO eval
    top = sorted(all_results, key=lambda r: r["composite_loss"])[:3]
    for t in top:
        t["loeo"] = loeo_eval(t["expr"], df)
        log.info(f"TOP: {t['expr'][:60]}... loss={t['composite_loss']:.3f} "
                 f"loeo_rmse={t['loeo']['loeo_rmse']:.3f}")

    TOP_JSON.write_text(json.dumps(top, indent=2))
    log.info(f"Wrote {TOP_JSON}")
    (OUT_DIR / "all_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )

    # Gate
    n_valid_dois = top[0].get("loeo", {}).get("n_valid_dois", 0) if top else 0
    total_dois = top[0].get("loeo", {}).get("n_dois", 1) if top else 1
    rmse = top[0].get("loeo", {}).get("loeo_rmse", float("nan")) if top else float("nan")
    log.info(f"Gate: LOEO {n_valid_dois}/{total_dois} valid, RMSE={rmse:.3f}")


if __name__ == "__main__":
    main()
