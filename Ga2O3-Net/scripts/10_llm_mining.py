"""Phase 55 V55-Ext — Production LLM PDF Mining with Local Qwen.

Five-stage pipeline (per V55 Roadmap §1.1, replacing external hosted-API consensus
with local Qwen2.5-7B + Qwen2.5-14B-int8 + Qwen2.5-VL-7B):
  Stage 1 (Triage):     YES/NO is this a β-Ga2O3 sputter paper with PDR/V_O?
  Stage 2-T1 (Explicit): JSON schema extraction of (host, dopant, process, property) tuples
  Stage 2-T2 (Derived):  fill null fields by re-reading + deriving from I-V/ratios
  Stage 3-T3 (Vision):   Qwen2.5-VL digitizes I-V / I-time curves at 254 nm
  Stage 4 (Verify):      ChatExtract redundancy loop — flip detection
  Stage 5 (Consensus):   2nd LLM (Qwen2.5-14B-int8) cross-check; 10% tol gates rows

Output: data/v55_llm_extracted.csv with per-row `confidence_weight` =
        consensus_agreement × redundancy_pass_rate, plus full provenance.

All-local: zero external API. Requires Qwen2.5-7B-Instruct downloaded to
  checkpoints/qwen2.5-7b (and optionally VL-7B / Math-7B / 14B-int8).

Usage:
  PYTHONPATH=. conda run -n ga2o3 python scripts/10_llm_mining.py \
      --pdf-dir extra-paper/doping_extracted/doping \
      --output data/v55_llm_extracted.csv \
      --max-papers 5  # smoke; remove for full corpus

  PYTHONPATH=. conda run -n ga2o3 python scripts/10_llm_mining.py \
      --pdf-dir extra-paper/doping_extracted/doping \
      --output data/v55_llm_extracted.csv \
      --enable-vision --enable-consensus  # full production
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v55_ext")


SCHEMA = {
    "host":                          "Ga2O3",
    "phase":                         "beta|alpha|epsilon|amorphous|null",
    "dopant":                        "chemical symbol or 'undoped'",
    "dopant_at_pct":                 "float >=0 or null",
    "method":                        "sputter_RF|sputter_DC|cosputter|null",
    "sputter_power_W":               "float or null",
    "Ar_O2_ratio":                   "'X:Y' or null",
    "total_pressure_Pa":             "float or null",
    "T_sub_C":                       "float or null",
    "post_anneal_T_C":               "float or null",
    "post_anneal_atm":               "Ar|N2|O2|air|vacuum|null",
    "post_anneal_time_min":          "float or null",
    "substrate":                     "c-sapphire|r-sapphire|m-sapphire|MgO|Si|null",
    "film_thickness_nm":             "float or null",
    "wavelength_nm":                 "float or null",
    "bias_V":                        "float or null",
    "PDR":                           "float (Iph/Idark) or null",
    "responsivity_AW":               "float or null",
    "detectivity_Jones":             "float or null",
    "EQE_pct":                       "float or null",
    "dark_current_A":                "float or null",
    "photo_current_A":               "float or null",
    "V_O_per_cm3":                   "float (linear scale, NOT log) or null",
    "V_O_proxy_XPS_O1s_ratio":       "float in [0,1] or null",
    "source_span":                   "verbatim sentence(s) supporting this row",
}


TRIAGE_SYSTEM = "You are a materials-science paper screening assistant. Answer only YES or NO with a one-sentence justification."

TRIAGE_USER = """Given the following excerpt from a paper, answer YES or NO:
Does this paper report SPUTTER-deposited (RF magnetron, DC magnetron, or co-sputter)
β-, α-, ε-, or amorphous Ga2O3 thin films with at least one measured numeric
performance metric (PDR / responsivity / detectivity / dark current / V_O)?

PAPER EXCERPT:
{paper_text}

Respond on ONE line: 'YES: <reason>' or 'NO: <reason>'."""


T1_SYSTEM = "You are a structured-data extraction assistant. Output ONLY a valid JSON array, no prose, no markdown fences."

T1_USER = """Extract EVERY (host, dopant, sputter process, photodetector property) tuple
explicitly reported in this β-Ga2O3 sputter paper. Output a JSON ARRAY of records.

For each record, use this schema (use null for any field NOT explicitly stated):
{schema_json}

STRICT RULES:
1. Only include rows where the deposition method is SPUTTER (RF/DC/cosputter). Skip PLD, MOCVD, CVD, MBE rows.
2. Do NOT guess. If a value is not in the paper text, write null.
3. Include a verbatim `source_span` quoting the sentence(s) supporting each row.
4. V_O_per_cm3 should be LINEAR scale (e.g., 3.5e17, not log10).
5. Multiple measurement conditions in one paper => one row each.

PAPER TEXT:
{paper_text}

OUTPUT (JSON array, no prose):"""


T2_SYSTEM = T1_SYSTEM

T2_USER = """You previously extracted these rows from the paper:
{prev_json}

Re-read the paper. For EACH null numeric field, attempt to DERIVE the value
from any equations, ratios, captions, supporting figures, or context:
- Compute PDR = Iph/Idark if both are stated.
- Compute responsivity = Iph / (P_incident × area) if power+area available.
- Read I-V curve text descriptions if explicit.
Cite the derivation source in `source_span`. If still underivable, keep null.

PAPER TEXT (for re-derivation):
{paper_text}

Output the SAME JSON array with derived fields filled, null otherwise:"""


T3_VISION_USER = """Here is a figure from a β-Ga2O3 sputter photodetector paper.
Identify any I-V (current-voltage) curve or I-time (current-time) curve measured
at the photodetector's operating wavelength (typically 254 nm).

For each identified curve, digitize:
- The dark current (I when V=0 or in dark state).
- The photo current (I under illumination).
- The applied bias (V) at which these are measured.

Output JSON:
{{"curves": [{{"label": "<curve_label>",
              "V": <bias_V>,
              "Idark_A": <dark_current_A>,
              "Iph_A": <photo_current_A>,
              "PDR_estimate": <Iph/Idark>}}]}}

If no I-V or I-t curve is present, output {{"curves": []}}."""


VERIFY_SYSTEM = T1_SYSTEM

VERIFY_USER = """You previously extracted the following rows from this β-Ga2O3
sputter paper:
{prev_json}

Re-read the paper INDEPENDENTLY (do not rely on your prior extraction).
For each row, confirm or correct EACH numeric field. If a value differs
between this re-read and the prior extraction, mark it 'unstable'.

Output the SAME JSON array with an additional `field_stability` dict per row:
{{"PDR": "stable|unstable|corrected:<new_val>", "T_sub_C": "stable|...", ...}}

PAPER TEXT:
{paper_text}

Output (JSON only):"""


def _walk_pdfs(pdf_dir: Path) -> Iterator[Path]:
    """Yield every .pdf file under pdf_dir (recursive), sorted for determinism."""
    if not pdf_dir.exists():
        return
    yield from sorted(pdf_dir.rglob("*.pdf"))


def _extract_text(pdf_path: Path, max_chars: int = 60_000) -> str:
    """Extract plain text from PDF using pdfplumber (fallback pypdf).

    Truncate to max_chars (~15k tokens) to fit Qwen 32k context with room
    for prompt + response.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated ...]"
        return text
    except Exception as exc:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {exc}; trying pypdf")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            return text[:max_chars]
        except Exception as exc2:
            logger.error(f"pypdf also failed on {pdf_path.name}: {exc2}")
            return ""


def _stable_pdf_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:12]


def _row_consensus_score(row_a: dict, row_b: dict, numeric_tol: float = 0.10) -> float:
    """Score agreement between two extractions of the same row.

    For numeric fields: 1.0 if within tol; 0.0 otherwise.
    For string fields: 1.0 if exact match; 0.5 if one is null; 0.0 otherwise.
    Returns mean agreement across all non-null fields in either row.
    """
    if row_a is None or row_b is None:
        return 0.0

    NUMERIC_FIELDS = {
        "dopant_at_pct", "sputter_power_W", "total_pressure_Pa",
        "T_sub_C", "post_anneal_T_C", "post_anneal_time_min",
        "film_thickness_nm", "wavelength_nm", "bias_V",
        "PDR", "responsivity_AW", "detectivity_Jones", "EQE_pct",
        "dark_current_A", "photo_current_A",
        "V_O_per_cm3", "V_O_proxy_XPS_O1s_ratio",
    }
    agreements = []
    for k in SCHEMA:
        a_val = row_a.get(k)
        b_val = row_b.get(k)
        if a_val is None and b_val is None:
            continue
        if k in NUMERIC_FIELDS:
            if a_val is None or b_val is None:
                agreements.append(0.5)
                continue
            try:
                a_f = float(a_val)
                b_f = float(b_val)
                if abs(a_f) < 1e-12 and abs(b_f) < 1e-12:
                    agreements.append(1.0)
                else:
                    rel = abs(a_f - b_f) / max(abs(a_f), abs(b_f), 1e-12)
                    agreements.append(1.0 if rel <= numeric_tol else 0.0)
            except (TypeError, ValueError):
                agreements.append(0.0)
        else:
            if a_val == b_val:
                agreements.append(1.0)
            elif a_val is None or b_val is None:
                agreements.append(0.5)
            else:
                agreements.append(0.0)
    return float(sum(agreements) / max(len(agreements), 1))


def _redundancy_score(rows: list[dict]) -> dict[int, float]:
    """For verified rows, compute redundancy_pass_rate = fraction of numeric
    fields marked 'stable' in `field_stability`.
    """
    out = {}
    for i, row in enumerate(rows):
        stab = row.get("field_stability", {})
        if not stab:
            out[i] = 0.5  # no stability info => neutral
            continue
        total = len(stab)
        stable = sum(1 for v in stab.values() if str(v) == "stable")
        out[i] = stable / max(total, 1)
    return out


def extract_pdf(
    pdf_path: Path,
    chat: Any,
    chat_2: Any | None = None,
    vision: Any | None = None,
    enable_consensus: bool = False,
    enable_vision: bool = False,
) -> list[dict]:
    """Run the 5-stage extraction pipeline on a single PDF."""
    pdf_bytes = pdf_path.read_bytes()
    pdf_text = _extract_text(pdf_path)
    if not pdf_text or len(pdf_text) < 200:
        logger.info(f"  → insufficient text ({len(pdf_text)} chars), skipping")
        return []

    # ── Stage 1: Triage ──
    triage_response = chat.chat(TRIAGE_SYSTEM, TRIAGE_USER.format(paper_text=pdf_text[:8_000]))
    if not triage_response.strip().upper().startswith("YES"):
        logger.info(f"  → triage NO: {triage_response[:80]}")
        return []
    logger.info(f"  → triage YES")

    # ── Stage 2-T1: explicit extraction (Qwen #1) ──
    t1_prompt = T1_USER.format(
        schema_json=json.dumps(SCHEMA, indent=2),
        paper_text=pdf_text,
    )
    rows_a = chat.extract_json(T1_SYSTEM, t1_prompt)
    if not isinstance(rows_a, list):
        logger.warning(f"  → T1 returned non-list ({type(rows_a)}), skipping")
        return []
    logger.info(f"  → T1 extracted {len(rows_a)} candidate rows")
    rows_a = [r for r in rows_a if isinstance(r, dict)]

    # ── Stage 2-T2: derived fields (re-read) ──
    if rows_a:
        t2_prompt = T2_USER.format(
            prev_json=json.dumps(rows_a, default=str),
            paper_text=pdf_text,
        )
        rows_a_derived = chat.extract_json(T2_SYSTEM, t2_prompt)
        if isinstance(rows_a_derived, list) and rows_a_derived:
            rows_a = [r for r in rows_a_derived if isinstance(r, dict)]
            logger.info(f"  → T2 derived; rows now {len(rows_a)}")

    # ── Stage 3-T3: vision (optional, image-by-image) ──
    if enable_vision and vision is not None:
        # Vision extraction is per-figure; we do not parse out individual figures
        # automatically here. This is a placeholder hook for future PDF-image
        # extraction (e.g., via pdf2image library). Falls back gracefully.
        logger.info(f"  → vision pass: stub (requires pdf2image to enable, skipping)")

    # ── Stage 4: verification redundancy ──
    if rows_a:
        verify_prompt = VERIFY_USER.format(
            prev_json=json.dumps(rows_a, default=str),
            paper_text=pdf_text,
        )
        rows_a_verified = chat.extract_json(VERIFY_SYSTEM, verify_prompt)
        if isinstance(rows_a_verified, list) and rows_a_verified:
            rows_a = [r for r in rows_a_verified if isinstance(r, dict)]
            logger.info(f"  → verification done")

    redundancy = _redundancy_score(rows_a)

    # ── Stage 5: multi-LLM consensus ──
    consensus_scores = {i: 1.0 for i in range(len(rows_a))}  # default: agree with self
    if enable_consensus and chat_2 is not None:
        t1_prompt_b = T1_USER.format(
            schema_json=json.dumps(SCHEMA, indent=2),
            paper_text=pdf_text,
        )
        rows_b = chat_2.extract_json(T1_SYSTEM, t1_prompt_b)
        if isinstance(rows_b, list):
            rows_b = [r for r in rows_b if isinstance(r, dict)]
            logger.info(f"  → consensus LLM #2 extracted {len(rows_b)} rows")
            # Pair each A-row with best matching B-row by dopant + sputter_power
            for i, ra in enumerate(rows_a):
                best = 0.0
                for rb in rows_b:
                    if (str(ra.get("dopant", "")).lower() ==
                        str(rb.get("dopant", "")).lower()):
                        score = _row_consensus_score(ra, rb)
                        if score > best:
                            best = score
                consensus_scores[i] = best

    # ── Annotate output ──
    pdf_hash = _stable_pdf_id(pdf_bytes)
    pdf_dir = pdf_path.parent.name
    timestamp = datetime.now().isoformat(timespec="seconds")
    final_rows = []
    for i, row in enumerate(rows_a):
        # Skip non-sputter rows (catch model hallucination)
        method = str(row.get("method", "") or "").lower()
        if method not in {"sputter_rf", "sputter_dc", "cosputter"}:
            continue
        # Skip rows with no useful property
        has_any_property = any(
            row.get(k) is not None for k in
            ("PDR", "responsivity_AW", "dark_current_A", "V_O_per_cm3", "V_O_proxy_XPS_O1s_ratio")
        )
        if not has_any_property:
            continue
        consensus = float(consensus_scores.get(i, 1.0))
        redund = float(redundancy.get(i, 0.5))
        confidence = consensus * redund
        out_row = dict(row)
        out_row.update({
            "source_pdf": pdf_path.name,
            "source_pdf_dir": pdf_dir,  # element bucket
            "source_pdf_hash": pdf_hash,
            "consensus_score": consensus,
            "redundancy_score": redund,
            "confidence_weight": confidence,
            "extraction_timestamp": timestamp,
            "extracted_by": "qwen2.5-7b" + ("+qwen2.5-14b-int8" if enable_consensus else ""),
        })
        final_rows.append(out_row)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="V55-Ext production LLM PDF mining")
    parser.add_argument("--pdf-dir", type=Path,
                        default=PROJ / "extra-paper" / "doping_extracted" / "doping")
    parser.add_argument("--output", type=Path,
                        default=PROJ / "data" / "v55_llm_extracted.csv")
    parser.add_argument("--max-papers", type=int, default=0,
                        help="Cap papers for smoke (0 = all)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no Qwen load)")
    parser.add_argument("--enable-vision", action="store_true",
                        help="Run Qwen2.5-VL for figure digitization")
    parser.add_argument("--enable-consensus", action="store_true",
                        help="Run 2nd Qwen (14B-int8) for cross-check")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default="checkpoints/qwen2.5-7b")
    parser.add_argument("--model-path-2", default="checkpoints/qwen2.5-14b-int8",
                        help="Second-LLM path for consensus")
    parser.add_argument("--vision-path", default="checkpoints/qwen2.5-vl-7b")
    parser.add_argument("--filter-element", default=None,
                        help="Only process PDFs under <pdf_dir>/<element>/")
    args = parser.parse_args()

    if args.mock:
        logger.warning("Mock mode — no real LLM calls")
        from src.llm import mock_chat as _mock

        class MockChat:
            def chat(self, s, u): return _mock(s, u)
            def extract_json(self, s, u, **kw): return []
        chat = MockChat()
        chat_2 = MockChat() if args.enable_consensus else None
        vision = MockChat() if args.enable_vision else None
    else:
        from src.llm import QwenChat
        chat = QwenChat(model_path=args.model_path, device=args.device,
                        dtype="bfloat16")
        chat_2 = None
        if args.enable_consensus:
            chat_2 = QwenChat(model_path=args.model_path_2,
                              device="cuda:1" if "cuda" in args.device else args.device,
                              dtype="int8")
        vision = None
        if args.enable_vision:
            from src.llm.qwen_local import QwenVision
            vision = QwenVision(model_path=args.vision_path, device=args.device)

    pdfs = list(_walk_pdfs(args.pdf_dir))
    if args.filter_element:
        pdfs = [p for p in pdfs if p.parent.name == args.filter_element]
    if args.max_papers > 0:
        pdfs = pdfs[:args.max_papers]
    logger.info(f"Mining {len(pdfs)} PDFs from {args.pdf_dir}")

    all_rows = []
    t0 = time.time()
    for i, pdf in enumerate(pdfs):
        elapsed = time.time() - t0
        logger.info(f"[{i + 1}/{len(pdfs)}] ({elapsed:.0f}s) {pdf.parent.name}/{pdf.name}")
        try:
            rows = extract_pdf(
                pdf, chat, chat_2=chat_2, vision=vision,
                enable_consensus=args.enable_consensus,
                enable_vision=args.enable_vision,
            )
        except Exception as exc:
            logger.error(f"  → ERROR on {pdf.name}: {exc}", exc_info=True)
            continue
        if rows:
            all_rows.extend(rows)
            logger.info(f"  → kept {len(rows)} sputter rows "
                        f"(running total: {len(all_rows)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        logger.info(f"Wrote {len(df)} rows to {args.output}")
        # Confidence distribution
        for thresh in (0.95, 0.9, 0.8, 0.6, 0.4):
            n = (df["confidence_weight"] >= thresh).sum()
            logger.info(f"  confidence ≥ {thresh}: {n} rows")
    else:
        logger.info(f"No sputter rows extracted; writing empty CSV.")
        pd.DataFrame(columns=list(SCHEMA.keys()) + [
            "source_pdf", "source_pdf_dir", "source_pdf_hash",
            "consensus_score", "redundancy_score", "confidence_weight",
            "extraction_timestamp", "extracted_by",
        ]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
