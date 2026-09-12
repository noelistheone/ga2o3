"""V56-Ext-2 — Qwen 5-stage Pydantic-validated PDF mining pipeline.

Bottleneck addressed: V55-Ext 5% yield (166 PDFs → 9 high-confidence rows).
Goal: 15-20% yield × ~1500 PDFs → 225-300 candidate rows, ≥40 V_O rows.

Stages (per paper passing Stage 1):
  1. Triage             — Qwen-7B  (GPU 1) — binary {sputter, PDR, V_O, dopant}
  2-T1. Extract         — Qwen-14B-AWQ (GPU 0) — Pydantic SputterRow schema
  2-T2. Derive          — Qwen-14B-AWQ — fill null fields from quoted source spans
  3. Vision (optional)  — Qwen-VL-7B   — figure digitization (PlotExtract)
  4. Verification       — Qwen-14B-AWQ — ChatExtract field-by-field "is this certainly X?"
  5. Consensus          — Qwen-7B re-extract + numeric agreement → confidence_weight

Phase 56 design rules:
  - Zero API (no external hosted-API keys)
  - Pydantic-first (vs Phase 55 regex)
  - Resumable (cache per-paper Stage 1+2+4 outputs to disk)
  - GPU 0 hosts Qwen-14B-AWQ (~10GB int4)
  - GPU 1 hosts Qwen-7B (~16GB bf16)
  - 14B and 7B can also co-reside on cuda:0 if cuda:1 needed for retrain

Outputs:
  - data/v56_llm_extracted.csv          — final extracted rows
  - data/processed/v56_ext2_cache/      — per-paper stage outputs (JSON)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import (
    LLMClient,
    make_client,
    TriageResult,
    SputterRow,
    FieldVerification,
    CONFIDENCE_TO_WEIGHT,
    confidence_to_sample_weight,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v56_ext2")


MARKDOWN_DIR = PROJ / "extra-paper" / "v56_markdown"
CACHE_DIR = PROJ / "data" / "processed" / "v56_ext2_cache"
OUTPUT_CSV = PROJ / "data" / "v56_llm_extracted.csv"


# ============================================================
# Prompts
# ============================================================


TRIAGE_SYSTEM = (
    "You are a materials-science classifier. Read a paper excerpt and "
    "classify it on four binary axes. Output ONLY valid JSON."
)

TRIAGE_USER_TMPL = """Paper text (first ~4000 chars):

{text}

Classify whether this paper:
1. is_sputter_Ga2O3: deposits β-Ga2O3 (gallium oxide) thin films by sputtering
2. has_PDR: reports photo-dark current ratio (or photo current and dark current separately)
3. has_V_O: reports oxygen vacancy concentration (or proxy like XPS O1s deconvolution)
4. has_dopant: studies a doped Ga2O3 film (vs undoped intrinsic)

Provide a one-sentence reason."""


EXTRACT_SYSTEM = (
    "You are a precise materials-science data extractor. Extract one row of "
    "experimental data from the paper text. If a field is not stated, return null. "
    "Be conservative: never guess. The `quote` field MUST be a verbatim sentence "
    "(or two) from the paper that supports the values. Output ONLY valid JSON."
)

EXTRACT_USER_TMPL = """Paper text (first ~12000 chars):

{text}

Extract one SputterRow for the primary doped film reported. If the paper
reports multiple doping concentrations, choose the one giving the BEST PDR
(or BEST V_O if no PDR). Fields:

- dopant: single element symbol (e.g. "Sn", "Mg", "Si", "Zn"); null if undoped
- dopant_at_pct: dopant atomic percent (0-20)
- deposition_method: one of {{"RF sputter", "DC sputter", "MOCVD", "HVPE", "PLD", "Mist-CVD", "sol-gel", "ALD", "PE-ALD", "other"}}
- substrate_temp_C: deposition substrate temperature (°C, 20-1200)
- O2_Ar_ratio: O2:Ar gas flow ratio (0-10; 0=pure Ar, 1=equal)
- total_pressure_Pa, sputter_power_W, film_thickness_nm: as reported
- anneal_T_C, anneal_atm, anneal_time_min: post-deposition anneal
- substrate: substrate material (e.g. "c-sapphire", "Si", "native bulk")
- wavelength_nm, bias_V: UV measurement conditions
- PDR_log10: log10(photo / dark current ratio), e.g. 4.5 means PDR=3.2e4
- photo_current_A, dark_current_A, responsivity_AW: as reported
- V_O_log10_cm3: log10(oxygen vacancy concentration / cm^-3), physical range 14-22
- V_O_method: one of {{"XPS O1s deconvolution", "Hall", "PL", "SIMS", "positron annihilation", "TSC", "EPR", "Raman", "other"}}
- confidence: "high" (>=3 fields directly cited), "medium" (some derived), "low" (mostly null or guessed)
- quote: verbatim sentence(s) from the paper supporting the row

Numerical fields MUST respect physical ranges (Pydantic will reject otherwise)."""


VERIFY_SYSTEM = (
    "You are a critical reviewer. For each claimed field-value pair, decide if "
    "the paper text directly supports it. Output JSON list of FieldVerification."
)


def _build_verify_prompt(row: SputterRow, text: str) -> str:
    """Build a Stage-4 verification user prompt for the 3-5 critical fields."""
    critical_fields = {
        "dopant": row.dopant,
        "dopant_at_pct": row.dopant_at_pct,
        "PDR_log10": row.PDR_log10,
        "V_O_log10_cm3": row.V_O_log10_cm3,
        "anneal_T_C": row.anneal_T_C,
    }
    field_lines = []
    for k, v in critical_fields.items():
        if v is None:
            continue
        field_lines.append(f"  - field={k}, claimed_value={v}")

    return f"""Paper text (~12000 chars):

{text}

For each of the following claimed fields, decide: is the paper text DIRECTLY
supporting the value? Answer "yes" only if the value can be read off the text
without inference. Reply "uncertain" if it requires derivation. Reply "no" if
the text contradicts or the field is fabricated.

Claimed fields:
{chr(10).join(field_lines)}

Output: a JSON list of FieldVerification objects, one per field, in the same order.
Schema per object: {{"field": str, "claimed_value": float|null, "claimed_value_str": str|null, "verdict": "yes"|"no"|"uncertain", "reason": str}}"""


# ============================================================
# Stage helpers
# ============================================================


@dataclass
class PaperState:
    paper_id: str
    markdown_path: Path
    text_full: str
    text_head4k: str
    text_head12k: str


def _load_paper(md_path: Path) -> PaperState:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    return PaperState(
        paper_id=md_path.stem,
        markdown_path=md_path,
        text_full=text,
        text_head4k=text[:4000],
        text_head12k=text[:12000],
    )


def _cache_path(paper_id: str, stage: str) -> Path:
    return CACHE_DIR / paper_id / f"{stage}.json"


def _load_cached(paper_id: str, stage: str):
    p = _cache_path(paper_id, stage)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _save_cached(paper_id: str, stage: str, data) -> None:
    p = _cache_path(paper_id, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    p.write_text(json.dumps(data, indent=2, default=str))


# ============================================================
# Stages
# ============================================================


def stage1_triage(paper: PaperState, llm: LLMClient) -> Optional[TriageResult]:
    cached = _load_cached(paper.paper_id, "stage1_triage")
    if cached is not None:
        try:
            return TriageResult.model_validate(cached)
        except Exception:
            pass

    user = TRIAGE_USER_TMPL.format(text=paper.text_head4k)
    result = llm.extract_structured(TRIAGE_SYSTEM, user, TriageResult, max_retries=2)
    if result is not None:
        _save_cached(paper.paper_id, "stage1_triage", result)
    return result


def stage2_extract(paper: PaperState, llm: LLMClient) -> Optional[SputterRow]:
    cached = _load_cached(paper.paper_id, "stage2_extract")
    if cached is not None:
        try:
            return SputterRow.model_validate(cached)
        except Exception:
            pass

    user = EXTRACT_USER_TMPL.format(text=paper.text_head12k)
    result = llm.extract_structured(EXTRACT_SYSTEM, user, SputterRow, max_retries=2)
    if result is not None:
        _save_cached(paper.paper_id, "stage2_extract", result)
    return result


def stage4_verify(paper: PaperState, row: SputterRow, llm: LLMClient) -> dict[str, str]:
    """Return {field: verdict} for critical fields. Drops fields with verdict='no'."""
    cached = _load_cached(paper.paper_id, "stage4_verify")
    if cached is not None:
        return cached

    user = _build_verify_prompt(row, paper.text_head12k)
    raw = llm.chat(VERIFY_SYSTEM, user)
    # parse JSON list manually
    from src.llm.qwen_local import _extract_first_json
    data = _extract_first_json(raw)
    verdicts: dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            f = item.get("field")
            v = item.get("verdict")
            if isinstance(f, str) and v in ("yes", "no", "uncertain"):
                verdicts[f] = v
    _save_cached(paper.paper_id, "stage4_verify", verdicts)
    return verdicts


def _row_consensus(row_a: SputterRow, row_b: SputterRow) -> float:
    """Agreement rate between two extractions, 0-1.

    For numeric fields: count as agree if within 10% relative or 0.3 absolute (log scale).
    For categorical: exact match.
    """
    if row_a is None or row_b is None:
        return 0.0
    fields_to_compare = [
        "dopant", "dopant_at_pct", "deposition_method", "substrate_temp_C",
        "O2_Ar_ratio", "anneal_T_C", "anneal_atm", "PDR_log10", "V_O_log10_cm3",
    ]
    agree = 0
    compared = 0
    for f in fields_to_compare:
        a = getattr(row_a, f, None)
        b = getattr(row_b, f, None)
        if a is None and b is None:
            continue  # both null = not informative
        compared += 1
        if a is None or b is None:
            continue  # one-sided null = disagree
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            tol = max(0.10 * abs(a), 0.3)
            if abs(a - b) <= tol:
                agree += 1
        else:
            if str(a).strip().lower() == str(b).strip().lower():
                agree += 1
    return agree / compared if compared > 0 else 0.0


def stage5_consensus(
    paper: PaperState,
    row_primary: SputterRow,
    llm_secondary: LLMClient,
) -> tuple[float, Optional[SputterRow]]:
    """Re-extract with a second LLM; return (agreement_rate, secondary_row)."""
    cached = _load_cached(paper.paper_id, "stage5_consensus")
    if cached is not None:
        secondary_data = cached.get("secondary_row")
        try:
            sec = SputterRow.model_validate(secondary_data) if secondary_data else None
        except Exception:
            sec = None
        return float(cached.get("agreement_rate", 0.0)), sec

    user = EXTRACT_USER_TMPL.format(text=paper.text_head12k)
    row_secondary = llm_secondary.extract_structured(
        EXTRACT_SYSTEM, user, SputterRow, max_retries=1
    )
    rate = _row_consensus(row_primary, row_secondary) if row_secondary else 0.0
    _save_cached(paper.paper_id, "stage5_consensus", {
        "agreement_rate": rate,
        "secondary_row": row_secondary.model_dump() if row_secondary else None,
    })
    return rate, row_secondary


# ============================================================
# Pipeline
# ============================================================


def process_paper(
    paper: PaperState,
    llm_triage: LLMClient,
    llm_extract: LLMClient,
    llm_secondary: LLMClient,
    do_consensus: bool,
) -> Optional[dict]:
    """Run full 5-stage pipeline on one paper. Returns final row dict or None."""
    # Stage 1
    triage = stage1_triage(paper, llm_triage)
    if triage is None:
        return None
    if not triage.passes():
        return None

    # Stage 2
    row = stage2_extract(paper, llm_extract)
    if row is None:
        return None

    # Stage 4 verification (drop fields verdict=no)
    verdicts = stage4_verify(paper, row, llm_extract)
    fields_no = sum(1 for v in verdicts.values() if v == "no")
    fields_uncertain = sum(1 for v in verdicts.values() if v == "uncertain")
    fields_yes = sum(1 for v in verdicts.values() if v == "yes")
    redundancy_pass_rate = fields_yes / max(1, len(verdicts))

    # If a critical field is 'no', null it out
    if verdicts.get("PDR_log10") == "no":
        row.PDR_log10 = None
    if verdicts.get("V_O_log10_cm3") == "no":
        row.V_O_log10_cm3 = None
    if verdicts.get("dopant_at_pct") == "no":
        row.dopant_at_pct = None

    # Stage 5 consensus
    # Secondary returns None when 7B can't produce a valid SputterRow.
    # We use a neutral floor (0.5) rather than zeroing out the primary's
    # confidence (7B failing != 14B being wrong; Phase 55 evidence shows 14B > 7B).
    agreement_rate = 1.0
    secondary_row = None
    if do_consensus:
        agreement_rate, secondary_row = stage5_consensus(paper, row, llm_secondary)
        if secondary_row is None:
            agreement_rate = 0.5  # neutral when secondary unable to extract

    # Final confidence weight = base_label × (0.5 + 0.3*redundancy + 0.2*agreement)
    # Rationale: base_label (LLM self-assessment) is the dominant signal; verification
    # and consensus are bonus modifiers. Multiplicative across all three (Roadmap's
    # design) zeroes out perfectly-good high-confidence rows when the weaker 7B
    # secondary fails to re-extract — that's secondary-model weakness, not data noise.
    base_weight = CONFIDENCE_TO_WEIGHT[row.confidence]
    quality_factor = 0.5 + 0.3 * redundancy_pass_rate + 0.2 * agreement_rate
    confidence_weight = min(1.0, base_weight * quality_factor)

    out = row.model_dump()
    out["source_paper_id"] = paper.paper_id
    out["redundancy_pass_rate"] = round(redundancy_pass_rate, 3)
    out["agreement_rate"] = round(agreement_rate, 3)
    out["confidence_weight"] = round(confidence_weight, 3)
    out["fields_verified_yes"] = fields_yes
    out["fields_verified_no"] = fields_no
    out["fields_verified_uncertain"] = fields_uncertain
    return out


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-Ext-2 5-stage extraction pipeline")
    ap.add_argument("--pilot", type=int, default=None,
                    help="process only first N markdowns (smoke / pilot)")
    ap.add_argument("--device-triage", default="cuda:1",
                    help="GPU for Qwen-7B (triage + Stage 5 secondary)")
    ap.add_argument("--device-extract", default="cuda:0",
                    help="GPU for Qwen-14B-AWQ (Stage 2/4 primary)")
    ap.add_argument("--no-consensus", action="store_true",
                    help="skip Stage 5 multi-LLM consensus (faster, less robust)")
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = ap.parse_args()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect markdowns
    mds = sorted(MARKDOWN_DIR.glob("*.md"))
    if args.pilot:
        mds = mds[:args.pilot]
    logger.info(f"Found {len(mds)} markdown files to process")

    # Load LLM clients (lazy — load on first use)
    logger.info(f"Triage LLM: Qwen-7B on {args.device_triage}")
    llm_triage = make_client("qwen-7b", device=args.device_triage,
                              dtype="bfloat16", max_new_tokens=512)
    logger.info(f"Extract LLM: Qwen-14B-AWQ on {args.device_extract}")
    llm_extract = make_client("qwen-14b-awq", device=args.device_extract,
                               max_new_tokens=2048)
    llm_secondary = llm_triage  # reuse 7B for consensus

    # Output CSV
    fields = [
        "source_paper_id", "dopant", "dopant_at_pct", "deposition_method",
        "substrate_temp_C", "O2_Ar_ratio", "total_pressure_Pa", "sputter_power_W",
        "film_thickness_nm", "anneal_T_C", "anneal_atm", "anneal_time_min",
        "substrate", "wavelength_nm", "bias_V", "PDR_log10",
        "photo_current_A", "dark_current_A", "responsivity_AW",
        "V_O_log10_cm3", "V_O_method", "confidence", "quote",
        "redundancy_pass_rate", "agreement_rate", "confidence_weight",
        "fields_verified_yes", "fields_verified_no", "fields_verified_uncertain",
    ]
    fout = open(args.output, "w", newline="")
    w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    w.writeheader()

    t0 = time.time()
    n_triage_pass = 0
    n_extract_ok = 0
    n_verified = 0
    n_high_conf = 0

    for i, md in enumerate(mds, 1):
        paper = _load_paper(md)
        if len(paper.text_full) < 500:
            continue
        try:
            row = process_paper(
                paper, llm_triage, llm_extract, llm_secondary,
                do_consensus=not args.no_consensus,
            )
        except Exception as e:
            logger.warning(f"  Paper {paper.paper_id}: {e}")
            continue

        # Stage 1 stats — re-load cache to count triage pass
        triage_data = _load_cached(paper.paper_id, "stage1_triage")
        if triage_data and triage_data.get("is_sputter_Ga2O3"):
            n_triage_pass += 1

        if row is None:
            if i % 20 == 0:
                logger.info(f"  [{i}/{len(mds)}] triage_pass={n_triage_pass} "
                            f"extract_ok={n_extract_ok} (skipped)")
            continue

        n_extract_ok += 1
        if row["confidence_weight"] >= 0.6:
            n_verified += 1
        if row["confidence_weight"] >= 0.9:
            n_high_conf += 1

        w.writerow(row)
        fout.flush()

        if i % 10 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(
                f"  [{i}/{len(mds)}] triage_pass={n_triage_pass} "
                f"extract_ok={n_extract_ok} verified_>=0.6={n_verified} "
                f"high_conf_>=0.9={n_high_conf} ({elapsed:.1f}m, "
                f"{elapsed*60/max(1,i):.1f}s/paper)"
            )

    fout.close()
    elapsed = (time.time() - t0) / 60
    yield_pct = 100 * n_extract_ok / max(1, len(mds))
    logger.info(
        f"DONE: {len(mds)} papers, triage_pass={n_triage_pass}, "
        f"extract_ok={n_extract_ok} (yield={yield_pct:.1f}%), "
        f"verified_>=0.6={n_verified}, high_conf_>=0.9={n_high_conf}, "
        f"elapsed={elapsed:.1f}m"
    )
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
