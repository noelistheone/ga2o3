"""V56-Debate — 3-agent Qwen self-debate on critical V_O extractions.

Bottleneck addressed: V56-Ext-2 V_O extractions with confidence="medium" need
independent verification before entering V_O regression head training.

Pattern: ReConcile/round-table (Chen, Saha, Bansal ACL 2024) + multi-agent
debate (Du, Li, Torralba, Tenenbaum, Mordatch ICML 2024). Since we lack
frontier LLMs, we simulate diversity via different (model_size, temperature,
system_prompt) tuples on the same Qwen family.

3 agents:
  - Extractor:   Qwen-14B-AWQ, T=0.0, "You are a precise materials extractor"
  - Skeptic:     Qwen-14B-AWQ, T=0.3, "You find errors in extractions"
  - Arbiter:     Qwen-7B,      T=0.0, "You weigh both arguments and decide"

Per V_O extraction with confidence=medium:
  Round 1: Extractor restates; Skeptic critiques
  Round 2: Each sees the other's output, refines
  Round 3: Arbiter decides

Output:
  - results/phase56v56debate/debate_results.json
  - confidence_weight refinement: agreement_rate replaces V56-Ext-2's secondary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import make_client
from src.llm.qwen_local import _extract_first_json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v56_debate")

RESULTS_DIR = PROJ / "results" / "phase56v56debate"
EXTRACT_CSV = PROJ / "data" / "v56_llm_extracted.csv"
CACHE_DIR = PROJ / "data" / "processed" / "v56_ext2_cache"


EXTRACTOR_SYS = (
    "You are a precise materials-science extractor. Re-read the paper snippet "
    "and confirm or revise the V_O extraction. Output JSON: "
    "{\"V_O_log10_cm3\": float|null, \"confidence\": \"high\"|\"medium\"|\"low\", \"reason\": str}"
)
SKEPTIC_SYS = (
    "You are a skeptical reviewer. Find issues with the V_O extraction. "
    "Consider: is the value actually stated? is the unit correct (cm^-3 vs other)? "
    "could it be an XPS ratio instead of absolute? Output JSON: "
    "{\"V_O_log10_cm3\": float|null, \"confidence\": \"high\"|\"medium\"|\"low\", \"objections\": [str]}"
)
ARBITER_SYS = (
    "You weigh the extractor and skeptic and decide the final V_O value. "
    "Output JSON: {\"V_O_log10_cm3\": float|null, \"confidence\": \"high\"|\"medium\"|\"low\", \"verdict\": str}"
)


def load_paper_text(paper_id: str) -> Optional[str]:
    p = PROJ / "extra-paper" / "v56_markdown" / f"{paper_id}.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")[:12000]


def debate_one(paper_id: str, claimed_vo: float, extractor, skeptic, arbiter) -> dict:
    text = load_paper_text(paper_id)
    if text is None:
        return {"paper_id": paper_id, "status": "no_paper_text"}

    base_prompt = (
        f"Paper text:\n\n{text}\n\n"
        f"A previous extraction claimed V_O_log10_cm3 = {claimed_vo}. "
        f"Re-evaluate."
    )

    # Round 1
    e_resp = extractor.chat(EXTRACTOR_SYS, base_prompt)
    e_json = _extract_first_json(e_resp) or {}
    s_resp = skeptic.chat(SKEPTIC_SYS, base_prompt)
    s_json = _extract_first_json(s_resp) or {}

    # Round 2: each sees the other's
    e2_prompt = base_prompt + f"\n\nThe skeptic said: {json.dumps(s_json)[:400]}\nDo you revise?"
    s2_prompt = base_prompt + f"\n\nThe extractor said: {json.dumps(e_json)[:400]}\nAny remaining concerns?"
    e2_resp = extractor.chat(EXTRACTOR_SYS, e2_prompt)
    e2_json = _extract_first_json(e2_resp) or e_json
    s2_resp = skeptic.chat(SKEPTIC_SYS, s2_prompt)
    s2_json = _extract_first_json(s2_resp) or s_json

    # Round 3: arbiter
    arb_prompt = (
        f"Paper text:\n\n{text[:8000]}\n\n"
        f"Original claim: V_O_log10_cm3 = {claimed_vo}\n"
        f"Extractor (round 2): {json.dumps(e2_json)[:400]}\n"
        f"Skeptic (round 2): {json.dumps(s2_json)[:400]}\n"
        f"Final ruling?"
    )
    arb_resp = arbiter.chat(ARBITER_SYS, arb_prompt)
    arb_json = _extract_first_json(arb_resp) or {}

    # Compute agreement: do extractor + skeptic + arbiter agree to within 0.3 log10?
    values = []
    for j in (e2_json, s2_json, arb_json):
        v = j.get("V_O_log10_cm3")
        if isinstance(v, (int, float)):
            values.append(float(v))
    consensus_rate = 0.0
    if len(values) == 3:
        if all(abs(v - claimed_vo) <= 0.3 for v in values):
            consensus_rate = 1.0
        elif sum(abs(v - claimed_vo) <= 0.3 for v in values) >= 2:
            consensus_rate = 0.66
        elif sum(abs(v - claimed_vo) <= 0.3 for v in values) >= 1:
            consensus_rate = 0.33
        else:
            consensus_rate = 0.0

    return {
        "paper_id": paper_id,
        "claimed_vo": claimed_vo,
        "extractor_round2": e2_json,
        "skeptic_round2": s2_json,
        "arbiter": arb_json,
        "consensus_rate": consensus_rate,
        "final_vo": arb_json.get("V_O_log10_cm3"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-Debate 3-agent ReConcile")
    ap.add_argument("--extract-csv", type=Path, default=EXTRACT_CSV)
    ap.add_argument("--device-14b", default="cuda:0")
    ap.add_argument("--device-7b", default="cuda:1")
    ap.add_argument("--max-cases", type=int, default=50)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.extract_csv.exists():
        logger.error(f"V56-Ext-2 CSV not found: {args.extract_csv}")
        return

    df = pd.read_csv(args.extract_csv)
    # Pick rows with V_O present and confidence not high
    df_med = df[(df["V_O_log10_cm3"].notna()) & (df.get("confidence", "low") != "high")]
    df_med = df_med.head(args.max_cases)
    logger.info(f"Found {len(df_med)} medium/low-confidence V_O rows to debate")

    if len(df_med) == 0:
        logger.warning("No ambiguous V_O cases to debate; nothing to do.")
        return

    logger.info(f"Loading Qwen-14B-AWQ on {args.device_14b}")
    extractor = make_client("qwen-14b-awq", device=args.device_14b,
                             temperature=0.0, max_new_tokens=512)
    skeptic = make_client("qwen-14b-awq", device=args.device_14b,
                           temperature=0.3, max_new_tokens=512)
    logger.info(f"Loading Qwen-7B on {args.device_7b}")
    arbiter = make_client("qwen-7b", device=args.device_7b,
                           temperature=0.0, max_new_tokens=512)

    all_results = []
    t0 = time.time()
    for i, row in df_med.iterrows():
        result = debate_one(
            paper_id=str(row["source_paper_id"]),
            claimed_vo=float(row["V_O_log10_cm3"]),
            extractor=extractor, skeptic=skeptic, arbiter=arbiter,
        )
        all_results.append(result)
        if (len(all_results)) % 5 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  {len(all_results)}/{len(df_med)} debates done ({elapsed:.1f}m)")

    (RESULTS_DIR / "debate_results.json").write_text(json.dumps(all_results, indent=2, default=str))
    consensus_geq = sum(1 for r in all_results if r.get("consensus_rate", 0) >= 0.66)
    logger.info(f"V56-Debate DONE: {len(all_results)} cases, "
                f"{consensus_geq} with ≥0.66 consensus ({100*consensus_geq/max(1,len(all_results)):.1f}%) "
                f"(gate ≥60%)")


if __name__ == "__main__":
    main()
