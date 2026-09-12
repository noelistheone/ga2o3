"""Phase 55 V55-Coscientist — Qwen Agent for Inverse Sputter Recipe Design.

ChemCrow-style (Bran et al., Nat. Mach. Intell. 6:525, 2024) local agent with
Qwen2.5-7B + V54 tools. Replaces external GPT-4-tools with local Qwen and a
deterministic Python tool dispatch.

Tools available to the agent:
  - v54c1_formula_eval(element):  evaluate V54-C1 closed-form on the dopant
  - lookup_dopant_property(elem): return EN, radius, ox_state, hardness, etc.
  - search_v55ext_corpus(elem):   fetch matching rows from V55-Ext extracted CSV
  - predict_v54_proxy(recipe):    approximate V_O prediction from C1 + heuristic

Inverse design target (V55 Roadmap §6.2):
  PDR > 10^5 AND V_O ∈ [10^17, 10^18] cm^-3
  Cover {Sn, Si, Hf, Sc, Mg, Cu+Bi co-dope}
  Output 6 sputter recipes with justification

Output: docs/phase55_inverse_design_recommendations.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_co")


# Tool implementations ---------------------------------------------------------

def lookup_dopant_property(element: str) -> dict:
    """Look up physics descriptors for a dopant element."""
    from src.data.element_descriptors import _RAW_DESCRIPTORS
    if element not in _RAW_DESCRIPTORS:
        return {"error": f"element {element} not in V52b descriptor table"}
    raw = _RAW_DESCRIPTORS[element]
    return {
        "EN": raw[0],
        "ionic_radius": raw[1],
        "carrier_type": raw[2],
        "delta_v_vs_Ga3+": raw[3],
        "delta_period_vs_row4": raw[4],
        "lone_pair_active": bool(raw[5]),
        "walsh_D": raw[6],
        "pearson_hardness": raw[7],
        "chi_minus_chi_Ga": raw[8],
        "radius_mismatch_vs_Ga": raw[9],
        "ox_state_offset": raw[10],
        "ox_state": raw[3] + 3,  # derived
    }


def v54c1_formula_eval(element: str) -> dict:
    """Evaluate V54-C1 closed-form V_O offset for the dopant."""
    p = lookup_dopant_property(element)
    if "error" in p:
        return p
    try:
        x2 = p["ox_state"]
        x3 = 1.0 if p["lone_pair_active"] else 0.0
        x4 = p["walsh_D"]
        x5 = p["pearson_hardness"]
        x6 = p["chi_minus_chi_Ga"]
        x7 = p["radius_mismatch_vs_Ga"]
        offset = (((x2 - x3) / math.exp(x5 * 0.09953339 + x7)) - x6) / 4.6604757 - x4 + 0.088079505
        return {
            "v54c1_v_o_offset": float(offset),
            "interpretation": (
                f"Predicted log10 V_O offset = {offset:+.3f}. "
                f"Positive = relatively higher V_O vs baseline; negative = suppresses V_O."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def search_v55ext_corpus(element: str, csv_path: Path | None = None) -> list[dict]:
    """Return V55-Ext extracted rows for the given dopant element."""
    if csv_path is None:
        csv_path = PROJ / "data" / "v55_llm_extracted.csv"
    if not csv_path.exists():
        return [{"warning": "V55-Ext CSV not yet generated"}]
    try:
        df = pd.read_csv(csv_path)
        if "dopant" not in df.columns:
            return [{"warning": "no dopant column"}]
        hits = df[df["dopant"].astype(str).str.contains(element, na=False, case=False)]
        return hits.to_dict(orient="records")[:5]  # cap
    except Exception as e:
        return [{"error": str(e)}]


def predict_v54_proxy(recipe: dict) -> dict:
    """Approximate V_O / PDR prediction using V54-C1 + heuristic process priors."""
    elem = recipe.get("dopant")
    p = lookup_dopant_property(elem)
    if "error" in p:
        return p
    c1 = v54c1_formula_eval(elem)
    if "error" in c1:
        return c1
    base_v_o = 17.5  # baseline log10 V_O cm^-3 for sputter undoped
    base_pdr = 3.0   # log10 PDR baseline
    v_o = base_v_o + c1["v54c1_v_o_offset"]
    # Heuristic process adjustments
    ar_o2 = recipe.get("Ar_O2_ratio", "9:1")
    try:
        ar, o2 = map(float, ar_o2.split(":"))
        o2_frac = o2 / (ar + o2)
    except (ValueError, AttributeError):
        o2_frac = 0.1
    # More O2 suppresses V_O
    v_o -= 0.5 * o2_frac
    # Anneal temperature: high T promotes V_O annihilation
    anneal_T = float(recipe.get("anneal_T_C", 0) or 0)
    if anneal_T >= 700:
        v_o -= 0.4
    # PDR heuristic: log10 PDR ≈ log10 PDR_undoped + slight dopant boost
    pdr = base_pdr + 0.5 * abs(c1["v54c1_v_o_offset"])
    # Uncertainty wider for unseen elements (Sc, Hf, Y, RE)
    n14_dopants = {"Mg", "Sn", "Si", "Zn", "Fe", "Cu", "Ge", "Bi", "Sb",
                   "Ti", "Ta", "Er", "Eu", "Al"}
    sigma = 0.3 if elem in n14_dopants else 0.6
    return {
        "log10_V_O_per_cm3": float(v_o),
        "log10_PDR": float(pdr),
        "sigma_log10": float(sigma),
        "v54c1_offset": float(c1["v54c1_v_o_offset"]),
    }


SYSTEM_PROMPT = """You are a β-Ga2O3 sputter-deposition expert. You design
sputter recipes for thin-film photodetectors to meet quantitative targets.

You have access to tool outputs (JSON dicts) for each candidate dopant:
  - Physical descriptors (electronegativity, ionic radius, oxidation state,
    Pauling hardness, Walsh-D off-centering, lone-pair flag).
  - V54-C1 closed-form V_O offset prediction.
  - V54 ensemble proxy prediction (log10 V_O + log10 PDR + sigma).
  - V55-Ext mined literature rows (if any).

Use these tools to design recipes that meet the user target.

OUTPUT FORMAT for each recipe (Markdown):
### Recipe N: <dopant> @ <c_at_pct>%
- **Sputter conditions**: <method>, <power_W>, Ar:O2 <ratio>, <pressure_Pa> Pa, T_sub <T_sub_C>°C
- **Post-anneal**: <T_C>°C / <atmosphere> / <time_min> min
- **Substrate**: <substrate>
- **Predicted V_O**: log10 = <value> ± <sigma> cm⁻³
- **Predicted PDR**: log10 = <value>
- **Justification** (2-3 sentences citing V54-C1 + descriptors + literature)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-Coscientist inverse design agent")
    parser.add_argument("--target-pdr-min", type=float, default=1e5)
    parser.add_argument("--vo-min-log10", type=float, default=17.0)
    parser.add_argument("--vo-max-log10", type=float, default=18.0)
    parser.add_argument("--candidates", default="Sn,Si,Hf,Sc,Mg,Cu+Bi,Bi,Cu,W,Al,Ti")
    parser.add_argument("--output", type=Path,
                        default=PROJ / "docs" / "phase55_inverse_design_recommendations.md")
    parser.add_argument("--qwen-model", default="checkpoints/qwen2.5-7b")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    candidates = [s.strip() for s in args.candidates.split(",")]
    logger.info(f"Designing recipes for {len(candidates)} candidate dopants: {candidates}")

    # Gather tool outputs for each candidate (deterministic, no LLM)
    tool_summary = {}
    for elem in candidates:
        # Handle co-dopants like "Cu+Bi" → use the dominant element heuristically
        if "+" in elem:
            primary = elem.split("+")[0].strip()
        else:
            primary = elem
        recipe_template = {
            "dopant": primary,
            "Ar_O2_ratio": "8:2",
            "anneal_T_C": 700,
        }
        tool_summary[elem] = {
            "props": lookup_dopant_property(primary),
            "v54c1": v54c1_formula_eval(primary),
            "v54_proxy": predict_v54_proxy(recipe_template),
            "literature": search_v55ext_corpus(primary),
        }

    # Build user prompt with tool outputs embedded
    user_prompt = f"""TARGET:
  - PDR (log10) > {math.log10(args.target_pdr_min):.1f}
  - V_O (log10 cm⁻³) ∈ [{args.vo_min_log10}, {args.vo_max_log10}]

CANDIDATES + TOOL OUTPUTS (use this data to design recipes):

{json.dumps(tool_summary, indent=2, default=str)[:6000]}

TASK: Design 6 sputter recipes (RF or DC magnetron) covering DIVERSE dopants
from the candidate list. Each recipe must be physically realistic in a standard
RF/DC sputter chamber. Justify with V54-C1 closed-form + tool predictions.

Cover at least: {{Sn, Si, Hf, Sc, Mg, Cu+Bi codope}}.

OUTPUT 6 recipes in the Markdown format shown in the system prompt.
"""

    from src.llm import QwenChat
    chat = QwenChat(model_path=args.qwen_model, device=args.device,
                    dtype="bfloat16", temperature=0.0, max_new_tokens=2048, seed=42)
    logger.info("Calling Qwen2.5-7B for inverse design...")
    response = chat.chat(SYSTEM_PROMPT, user_prompt, max_new_tokens=2048)
    logger.info(f"Got {len(response)} chars response")

    # Build the final markdown doc
    md = []
    md.append("# Phase 55 V55-Coscientist — Inverse Sputter Recipe Design")
    md.append(f"\n**Date**: 2026-05-23")
    md.append(f"**Agent**: Local Qwen2.5-7B-Instruct with V54 tools")
    md.append(f"**Target**: PDR > {args.target_pdr_min:.0e} AND V_O ∈ [10^{args.vo_min_log10:.0f}, 10^{args.vo_max_log10:.0f}] cm⁻³")
    md.append("\n## Candidate Dopants Tool Summary\n")
    for elem, tools in tool_summary.items():
        c1 = tools.get("v54c1", {})
        proxy = tools.get("v54_proxy", {})
        if "error" in c1:
            md.append(f"\n### {elem}\n  Skipped: {c1['error']}")
            continue
        md.append(f"\n### {elem}")
        props = tools["props"]
        md.append(f"- EN={props['EN']:.2f}, radius={props['ionic_radius']:.2f} Å, "
                  f"ox_state={props['ox_state']}, hardness={props['pearson_hardness']:.1f}, "
                  f"lone_pair={props['lone_pair_active']}, walsh_D={props['walsh_D']:.2f}")
        md.append(f"- V54-C1 V_O offset = {c1.get('v54c1_v_o_offset', float('nan')):+.3f} log10")
        md.append(f"- V54 proxy: log10 V_O ≈ {proxy.get('log10_V_O_per_cm3', float('nan')):.2f} ± "
                  f"{proxy.get('sigma_log10', float('nan')):.2f}, log10 PDR ≈ {proxy.get('log10_PDR', float('nan')):.2f}")
        lit = tools.get("literature", [])
        if lit and isinstance(lit, list) and lit and "warning" not in str(lit[0]):
            md.append(f"- V55-Ext literature hits: {len(lit)} rows")
        else:
            md.append("- V55-Ext literature: no specific match")

    md.append("\n## Agent Recipe Recommendations\n")
    md.append(response)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(md))
    logger.info(f"Wrote {args.output}")
    print(f"\n=== Tool-summary preview ===")
    for elem, tools in list(tool_summary.items())[:3]:
        c1 = tools.get("v54c1", {})
        proxy = tools.get("v54_proxy", {})
        if "error" not in c1:
            print(f"  {elem}: V54-C1 offset {c1.get('v54c1_v_o_offset', float('nan')):+.3f}, "
                  f"proxy log10 V_O {proxy.get('log10_V_O_per_cm3', float('nan')):.2f}")


if __name__ == "__main__":
    main()
