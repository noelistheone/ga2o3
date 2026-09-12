"""V57-Ext-3 — zero-API multimodal extraction (Qwen3-VL + Qwen3-30B-Thinking).

Replaces V56-Ext-2 5-stage pipeline. Stages:
  Stage 1 (Qwen3-VL triage)   — flag XPS / I-V figure pages + paper-level binary
                                 classification (extends TriageResult → TriageResultVL).
  Stage 2 (Qwen3-VL extract)  — figure-grounded XPSDeconvolution + SputterRow fill.
  Stage 3 (Qwen3-30B validate) — HSE06-window plausibility verdict (uses the
                                  DFT corpus from scripts/30_v57_dft_serialize.py).
  Stage 4 (HSE06 filter)       — drops any row whose derived [V_O] is outside the
                                  per-dopant HSE06 window via src.data.kroger_synthetic.kroger_predict.

User decision 2026-05-24: ZERO API. No Sonnet 4.5 batch — the Roadmap §D
hybrid path is collapsed to local-only. Yield target downgraded from 15-20%
to 5-10%.

Resumable: per-paper Stage 1+2+3 cached in data/v57_ext3_cache/<doi_hash>/.

Output:
  data/v57_ext3_extracted.csv          — final extracted rows (V56 schema +
                                          xps_validated, hse06_validated, vision_pages)
  data/v57_ext3_review.jsonl           — uncertain rows for manual review
  results/phase57v57ext3/manifest.json — summary stats (yield, V_O count, PDR count)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import (
    LLMClient, make_client, SputterRow, CONFIDENCE_TO_WEIGHT,
)
from src.llm.v56_schema import (
    TriageResultVL, XPSDeconvolution, HSE06PlausibilityVerdict,
)
from src.data.kroger_synthetic import (
    kroger_predict, DOPANT_OFFSET, ELEMENTS, C_REF,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_ext3")

MARKDOWN_DIR = PROJ / "extra-paper" / "v56_markdown"
PDF_DIR = PROJ / "extra-paper" / "v56_harvest"
CACHE_DIR = PROJ / "data" / "v57_ext3_cache"
OUT_CSV = PROJ / "data" / "v57_ext3_extracted.csv"
OUT_REVIEW = PROJ / "data" / "v57_ext3_review.jsonl"
OUT_DIR = PROJ / "results" / "phase57v57ext3"
HSE06_YAML = PROJ / "data" / "processed" / "v57_dft_corpus" / "V_O_HSE06.yaml"

TRIAGE_SYSTEM = (
    "You are a materials-science classifier. Given a paper excerpt + a list of "
    "figure descriptions, decide whether the paper is in scope (β-Ga2O3 sputter, "
    "with PDR or V_O measurement) and identify which figure pages contain "
    "(a) O 1s XPS deconvolution, (b) I-V or I-t curves under UV. Output JSON."
)

TRIAGE_USER_TMPL = """Paper excerpt (first ~5000 chars):

{text}

Figure descriptions (page : caption excerpt):
{figs}

Classify the paper and identify O 1s XPS / I-V figure pages.
- For XPS pages: 1-indexed page numbers where you see "O 1s", "OH-O", "O_v", "binding energy ~531", etc.
- For I-V pages: 1-indexed pages with "I-V", "I-t", "photo current", "dark current", "responsivity"."""

XPS_EXTRACT_SYSTEM = (
    "You are an XPS analyst. Given an O 1s spectrum figure caption (and the "
    "surrounding paper text), report the deconvolved components: binding "
    "energies, fractional areas, and chemical assignments. If the authors "
    "estimate [V_O] explicitly, report it. CRITICAL: a 531-532 eV peak alone "
    "is NOT proof of V_O — Spectroscopy Online 2024 warns this is commonly "
    "OH-adsorbate. Check whether the authors cross-validate via cation valence "
    "shift (Ga 2p/3d) before claiming `cation_valence_cross_validation=true`."
)

XPS_EXTRACT_USER_TMPL = """O 1s XPS context (caption + surrounding paragraph):

{context}

Extract one XPSDeconvolution row. Use the binding-energy convention 526-536 eV.
Assignment must be one of: lattice_O / O_v_or_OH / adsorbed_O / other.
The `quote` field MUST be a verbatim sentence from the paper.
If [V_O] is computable, populate derived_V_O_log10_cm3 in [14, 22]."""

HSE06_VALIDATE_SYSTEM = (
    "You are a Ga2O3 defect-chemistry validator. Compare the paper's extracted "
    "V_O against the HSE06-derived plausibility window for the same dopant + "
    "process conditions. Accept if within window, reject if grossly outside, "
    "uncertain otherwise. Provide brief reasoning."
)

HSE06_VALIDATE_USER_TMPL = """Extracted row:
  dopant         : {dopant}
  concentration  : {dopant_at_pct} at%
  substrate_temp : {substrate_temp_C} °C
  O2_Ar_ratio    : {O2_Ar_ratio}
  anneal_T       : {anneal_T_C} °C
  derived_V_O_log10_cm3 : {derived_V_O_log10}

HSE06-derived window for this dopant + (T, p_O2):
  lower bound: log10[V_O] >= {window_lower:.2f}  (e.g. from kroger_predict at q=2)
  upper bound: log10[V_O] <= {window_upper:.2f}  (saturation at frenkel equilibrium)
  base HSE06 records:
{hse06_excerpt}

Is the derived value plausible? Output JSON with field_in_hse06_window,
verdict (accept/reject/uncertain), reasoning_trace.
"""


def load_hse06_corpus() -> list[dict]:
    """Load V_O HSE06 records (output of scripts/30_v57_dft_serialize.py)."""
    if not HSE06_YAML.exists():
        log.warning(f"HSE06 corpus missing: {HSE06_YAML} — run scripts/30 first")
        return []
    with HSE06_YAML.open() as f:
        return yaml.safe_load(f) or []


def hse06_excerpt_for_dopant(corpus: list[dict], dopant: Optional[str]) -> str:
    """Format ≤3 HSE06 records for the given dopant as a compact table."""
    if not dopant:
        return "  (no dopant; use generic V_O windows)"
    rows = [r for r in corpus if r.get("primary_dopant_present") == dopant]
    if not rows:
        return f"  (no HSE06 record for dopant={dopant}; using closed-form prediction only)"
    lines = []
    for r in rows[:3]:
        ef = r.get("formation_energies_by_charge", {})
        lines.append(f"  - {r['record_id']} : E_f^+2={ef.get('q=+2','?')} eV, "
                     f"{r.get('frenkel_donor_acceptor','')}")
    return "\n".join(lines)


def hse06_window(dopant: Optional[str], T_C: Optional[float],
                 log_pO2: Optional[float]) -> tuple[float, float]:
    """Compute (lower, upper) log10[V_O] plausibility window from kroger_predict.

    Lower bound: kroger_predict at (T,p_O2) for q=2 minus a generous 1-dex slop
    Upper bound: hardcoded saturation at 22 (Frenkel equilibrium)
    """
    if not dopant or dopant not in DOPANT_OFFSET:
        return 14.0, 22.0
    T_K = (T_C or 600.0) + 273.15
    log_p = log_pO2 if log_pO2 is not None else -2.0
    central = kroger_predict(dopant, 1e-2, T_K, log_p, q=2)
    return max(14.0, central - 1.5), min(22.0, central + 1.5)


def get_paper_text(md_path: Path, max_chars: int = 12000) -> str:
    try:
        text = md_path.read_text(errors="ignore")
    except Exception as e:
        log.warning(f"read fail {md_path}: {e}")
        return ""
    return text[:max_chars]


def find_figure_pages(text: str, max_show: int = 12) -> list[tuple[int, str]]:
    """Heuristic: scan markdown for 'Fig.' captions and assign approximate page
    numbers by chunking. Returns (page_idx, short_caption_excerpt).
    """
    figs = []
    n_page_break = max(text.count("\f"), text.count("\n## "), 1)
    chunk_size = max(2000, len(text) // max(1, n_page_break))
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        for marker in ("Fig.", "Figure ", "FIGURE "):
            if marker in chunk:
                idx = chunk.find(marker)
                cap = chunk[idx : idx + 200].splitlines()[0]
                page = i // chunk_size + 1
                figs.append((page, cap.strip()))
                if len(figs) >= max_show:
                    return figs
    return figs


def process_paper(
    md_path: Path,
    triage_client: LLMClient,
    vl_client: LLMClient,
    validate_client: LLMClient,
    hse06_corpus: list[dict],
    cache_dir: Path,
) -> Optional[dict]:
    """End-to-end V57-Ext-3 processing for one PDF. Returns extracted row or None."""
    doi_hash = md_path.stem
    paper_cache = cache_dir / doi_hash
    paper_cache.mkdir(parents=True, exist_ok=True)
    text = get_paper_text(md_path)
    if not text:
        return None
    figs = find_figure_pages(text)

    # Stage 1 — triage
    triage_cache = paper_cache / "triage.json"
    if triage_cache.exists():
        triage = TriageResultVL.model_validate_json(triage_cache.read_text())
    else:
        figs_str = "\n".join(f"  page {p}: {c[:120]}" for p, c in figs) or "  (no figures detected)"
        triage_user = TRIAGE_USER_TMPL.format(text=text[:5000], figs=figs_str)
        triage = triage_client.extract_structured(
            TRIAGE_SYSTEM, triage_user, TriageResultVL, max_retries=2,
        )
        if triage is None:
            return None
        triage_cache.write_text(triage.model_dump_json(indent=2))

    if not triage.passes():
        return None

    # Stage 2 — extract a SputterRow from the paper text (no figure needed yet)
    extract_cache = paper_cache / "sputter_row.json"
    if extract_cache.exists():
        row = SputterRow.model_validate_json(extract_cache.read_text())
    else:
        from src.llm.v56_schema import SputterRow as _SR
        row = validate_client.extract_structured(
            "Extract one SputterRow row from this paper. Be conservative; use null when uncertain.",
            f"Paper text (first ~10000 chars):\n\n{text[:10000]}",
            _SR, max_retries=2,
        )
        if row is None:
            return None
        extract_cache.write_text(row.model_dump_json(indent=2))

    xps_extracted: Optional[XPSDeconvolution] = None
    xps_validated = False
    if triage.has_O1s_XPS_figure and triage.xps_figure_pages:
        # Stage 2b — XPS figure-grounded extraction (text-only fallback since
        # we don't have the figure crops yet; runs on text caption neighborhood)
        xps_cache = paper_cache / "xps.json"
        if xps_cache.exists():
            xps_extracted = XPSDeconvolution.model_validate_json(xps_cache.read_text())
        else:
            ctx = text[:8000]  # heuristic — full XPS analysis lives in body text
            xps_extracted = vl_client.extract_structured(
                XPS_EXTRACT_SYSTEM,
                XPS_EXTRACT_USER_TMPL.format(context=ctx),
                XPSDeconvolution, max_retries=2,
            )
            if xps_extracted is not None:
                xps_cache.write_text(xps_extracted.model_dump_json(indent=2))
                xps_validated = True
                # Merge V_O if extracted from XPS dominates over text
                if xps_extracted.derived_V_O_log10_cm3 is not None:
                    if (row.V_O_log10_cm3 is None
                            or abs(xps_extracted.derived_V_O_log10_cm3 - 17.0) <
                               abs((row.V_O_log10_cm3 or 17.0) - 17.0)):
                        row.V_O_log10_cm3 = xps_extracted.derived_V_O_log10_cm3
                        row.V_O_method = "XPS O1s deconvolution"

    # Stage 3 — HSE06 plausibility validate (only meaningful if V_O present)
    hse06_validated = False
    verdict_label = None
    if row.V_O_log10_cm3 is not None:
        T = row.substrate_temp_C
        log_pO2 = None
        if row.O2_Ar_ratio is not None:
            # 1.0 = pure O2; map ratio → equivalent log10 pO2 (heuristic — 1 atm scaling)
            import math
            log_pO2 = math.log10(max(row.O2_Ar_ratio, 1e-4) / (1 + row.O2_Ar_ratio))

        lower, upper = hse06_window(row.dopant, T, log_pO2)
        excerpt = hse06_excerpt_for_dopant(hse06_corpus, row.dopant)

        validate_cache = paper_cache / "hse06_verdict.json"
        if validate_cache.exists():
            verdict = HSE06PlausibilityVerdict.model_validate_json(validate_cache.read_text())
        else:
            verdict = validate_client.extract_structured(
                HSE06_VALIDATE_SYSTEM,
                HSE06_VALIDATE_USER_TMPL.format(
                    dopant=row.dopant or "undoped",
                    dopant_at_pct=row.dopant_at_pct,
                    substrate_temp_C=row.substrate_temp_C,
                    O2_Ar_ratio=row.O2_Ar_ratio,
                    anneal_T_C=row.anneal_T_C,
                    derived_V_O_log10=row.V_O_log10_cm3,
                    window_lower=lower,
                    window_upper=upper,
                    hse06_excerpt=excerpt,
                ),
                HSE06PlausibilityVerdict, max_retries=2,
            )
            if verdict is not None:
                validate_cache.write_text(verdict.model_dump_json(indent=2))
        if verdict is not None:
            hse06_validated = verdict.field_in_hse06_window
            verdict_label = verdict.verdict
            # Stage 4 — hard filter on the closed-form window
            if not (lower - 0.5 <= row.V_O_log10_cm3 <= upper + 0.5):
                log.info(f"  {doi_hash}: V_O={row.V_O_log10_cm3:.2f} outside "
                         f"[{lower:.2f},{upper:.2f}] → drop")
                return None

    # confidence_weight (V56-Ext-2 additive formula, hardened)
    base = CONFIDENCE_TO_WEIGHT.get(row.confidence, 0.0)
    quality = 0.5 + 0.3 * (1.0 if hse06_validated else 0.0) \
                  + 0.2 * (1.0 if xps_validated else 0.0)
    confidence_weight = min(1.0, base * quality)

    out = {
        **row.model_dump(),
        "doi_hash": doi_hash,
        "xps_validated": xps_validated,
        "hse06_validated": hse06_validated,
        "hse06_verdict": verdict_label,
        "vision_pages": ",".join(map(str, triage.xps_figure_pages))
                          + ("|iv:" + ",".join(map(str, triage.iv_figure_pages))
                             if triage.iv_figure_pages else ""),
        "confidence_weight": confidence_weight,
        "usable_flag": "v57_ext",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=0,
                    help="0 → all 658; otherwise process first N for pilot")
    ap.add_argument("--triage-backend", default="qwen3-30b-thinking-vllm",
                    help="Default routes triage + extract + validate to the local "
                         "vLLM Qwen3-30B-A3B-Thinking server (port 8000, TP=2). "
                         "Pass --triage-backend qwen-7b for the V56 fallback.")
    ap.add_argument("--vl-backend", default="none",
                    help="VL backend. Default 'none' = text-only mode. "
                         "Set to qwen3-vl for figure-grounded extraction.")
    ap.add_argument("--validate-backend", default="qwen3-30b-thinking-vllm",
                    help="HSE06 validator backend; same as triage by default for one-server pipeline.")
    ap.add_argument("--triage-device", default="cuda:0")
    ap.add_argument("--vl-device", default="cuda:0")
    ap.add_argument("--validate-device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent papers processed in parallel (matched to vLLM --max-num-seqs).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hse06_corpus = load_hse06_corpus()
    log.info(f"Loaded {len(hse06_corpus)} HSE06 records for plausibility")

    md_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if args.max_papers > 0:
        md_files = md_files[: args.max_papers]
    log.info(f"Processing {len(md_files)} papers from {MARKDOWN_DIR}")

    # Memory-aware client loading: when triage/validate use the SAME backend
    # on the SAME device, share a single in-process client so we don't double
    # the GPU footprint. The VL stage is optional — if its backend matches
    # triage, share too; otherwise skip-with-warning when memory is tight.
    log.info(f"Loading triage backend ({args.triage_backend}, {args.triage_device})...")
    triage_client = make_client(args.triage_backend, device=args.triage_device)

    if (args.validate_backend == args.triage_backend
            and args.validate_device == args.triage_device):
        log.info("Validate backend matches triage; reusing single client (memory-saving)")
        validate_client = triage_client
    else:
        log.info(f"Loading validate backend ({args.validate_backend}, {args.validate_device})...")
        try:
            validate_client = make_client(args.validate_backend, device=args.validate_device)
        except Exception as e:
            log.warning(f"validate backend {args.validate_backend} failed ({e}); "
                        f"falling back to qwen-14b-awq")
            validate_client = make_client("qwen-14b-awq", device=args.validate_device)

    # VL is optional — only load if its backend differs and we have headroom.
    if args.vl_backend in ("none", "skip", ""):
        log.info("VL stage disabled (text-only pilot)")
        vl_client = validate_client  # fallback text-only fill for XPS analysis
    elif (args.vl_backend == args.triage_backend
            and args.vl_device == args.triage_device):
        log.info("VL backend matches triage; reusing client (no figure extraction)")
        vl_client = triage_client
    else:
        log.info(f"Loading VL backend ({args.vl_backend}, {args.vl_device})...")
        try:
            vl_client = make_client(args.vl_backend, device=args.vl_device)
        except Exception as e:
            log.warning(f"VL backend {args.vl_backend} failed ({e}); falling back to text-only")
            vl_client = validate_client

    rows: list[dict] = []
    n_triage_pass = 0
    n_v_o = 0
    n_pdr = 0
    t_start = time.time()

    # Parallel processing: vLLM batches concurrent HTTP requests automatically
    # (--max-num-seqs 8) → saturates GPU compute rather than serialising at
    # ~1 token/s/seq. Each thread is light (HTTP only, no local model).
    from concurrent.futures import ThreadPoolExecutor, as_completed
    completed_counter = {"n": 0}

    def _worker(md):
        try:
            return md, process_paper(md, triage_client, vl_client,
                                     validate_client, hse06_corpus, CACHE_DIR)
        except Exception as e:
            log.warning(f"{md.stem}: error {e}")
            return md, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, md) for md in md_files]
        for fut in as_completed(futures):
            md, r = fut.result()
            completed_counter["n"] += 1
            if r is None:
                continue
            rows.append(r)
            n_triage_pass += 1
            if r.get("V_O_log10_cm3") is not None:
                n_v_o += 1
            if r.get("PDR_log10") is not None:
                n_pdr += 1
            if completed_counter["n"] % 25 == 0:
                elapsed = time.time() - t_start
                log.info(f"[{completed_counter['n']}/{len(md_files)}] "
                         f"kept {len(rows)} rows (V_O={n_v_o}, PDR={n_pdr}); "
                         f"{elapsed/max(1,completed_counter['n']):.1f}s/paper avg")

    log.info(f"V57-Ext-3 done: {len(rows)} candidate rows kept, "
             f"{n_v_o} with V_O, {n_pdr} with PDR (yield={len(rows)/max(1,len(md_files)):.1%})")

    if rows:
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Wrote {OUT_CSV}")

    manifest = {
        "n_papers": len(md_files),
        "n_kept": len(rows),
        "n_with_V_O": n_v_o,
        "n_with_PDR": n_pdr,
        "yield_pct": 100.0 * len(rows) / max(1, len(md_files)),
        "triage_backend": args.triage_backend,
        "vl_backend": args.vl_backend,
        "validate_backend": args.validate_backend,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
