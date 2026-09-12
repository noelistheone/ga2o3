"""V56-Ext-2 Stage 0.4 — parse PDFs to markdown.

Sources:
  - extra-paper/v56_harvest/*.pdf       (harvested by 20_v56_harvest_pdfs.py)
  - extra-paper/doping_extracted/doping/<element>/*.pdf  (Phase 55 corpus)

Backends:
  - pdfplumber (default): fast, no model download, V55-validated. Good for text;
    weaker on dense tables. Falls back to pypdf on pdfplumber failure.
  - mineru (--backend mineru): better table extraction (OmniDocBench 84.9 Tables),
    but downloads ~2GB layout-detection models on first use. Run as subprocess.

Output:
  - extra-paper/v56_markdown/<doi_hash>.md          markdown body
  - extra-paper/v56_markdown/manifest.csv           (doi_hash, source_pdf, n_chars, n_tables, status)

Resumable: skips PDFs already parsed (markdown file exists).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v56_parse")


HARVEST_DIR = PROJ / "extra-paper" / "v56_harvest"
LEGACY_DIR = PROJ / "extra-paper" / "doping_extracted" / "doping"
OUTPUT_DIR = PROJ / "extra-paper" / "v56_markdown"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"


def _doi_hash(path: Path) -> str:
    """8-char hash of normalized PDF path (stable identifier for output)."""
    return hashlib.sha256(str(path.name).encode("utf-8")).hexdigest()[:16]


def _collect_pdfs() -> list[Path]:
    """Find all PDFs in harvest + legacy dirs."""
    out: list[Path] = []
    if HARVEST_DIR.exists():
        out.extend(sorted(HARVEST_DIR.glob("*.pdf")))
    if LEGACY_DIR.exists():
        out.extend(sorted(LEGACY_DIR.glob("**/*.pdf")))
    return out


# ============================================================
# pdfplumber backend
# ============================================================


def parse_pdfplumber(pdf_path: Path) -> tuple[str, int, int]:
    """Parse PDF with pdfplumber. Returns (markdown, n_chars, n_tables)."""
    import pdfplumber

    pages_md: list[str] = []
    n_tables = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            txt = page.extract_text() or ""
            page_md = [f"## Page {page_num}\n", txt.strip()]

            # Extract tables as pipe-format markdown
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables, 1):
                if not table or not any(any(c for c in row) for row in table):
                    continue
                n_tables += 1
                header = table[0] if table[0] else [f"col{i}" for i in range(len(table[1] or []))]
                page_md.append(f"\n### Table {page_num}.{t_idx}\n")
                page_md.append("| " + " | ".join(str(c or "").strip() for c in header) + " |")
                page_md.append("| " + " | ".join("---" for _ in header) + " |")
                for row in table[1:]:
                    if not any(row):
                        continue
                    page_md.append("| " + " | ".join(str(c or "").strip() for c in row) + " |")
            pages_md.append("\n".join(page_md))

    md = "\n\n---\n\n".join(pages_md)
    return md, len(md), n_tables


def parse_pypdf_fallback(pdf_path: Path) -> tuple[str, int, int]:
    """Fallback to pypdf for malformed PDFs that pdfplumber can't open."""
    from pypdf import PdfReader

    pages_md: list[str] = []
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages_md.append(f"## Page {page_num}\n\n{txt.strip()}")
    md = "\n\n---\n\n".join(pages_md)
    return md, len(md), 0


def parse_one_pdfplumber(pdf_path_str: str, output_dir_str: str) -> dict:
    """Worker function for ProcessPoolExecutor."""
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)
    dh = _doi_hash(pdf_path)
    md_path = output_dir / f"{dh}.md"

    if md_path.exists() and md_path.stat().st_size > 100:
        return {
            "doi_hash": dh,
            "source_pdf": str(pdf_path.relative_to(PROJ)),
            "n_chars": md_path.stat().st_size,
            "n_tables": 0,
            "status": "skip_exists",
            "backend": "cache",
        }

    backend = "pdfplumber"
    try:
        md, n_chars, n_tables = parse_pdfplumber(pdf_path)
    except Exception as e:
        try:
            md, n_chars, n_tables = parse_pypdf_fallback(pdf_path)
            backend = "pypdf"
        except Exception as e2:
            return {
                "doi_hash": dh,
                "source_pdf": str(pdf_path.relative_to(PROJ)),
                "n_chars": 0,
                "n_tables": 0,
                "status": f"fail: {str(e)[:80]} | {str(e2)[:80]}",
                "backend": "none",
            }

    if n_chars < 500:
        return {
            "doi_hash": dh,
            "source_pdf": str(pdf_path.relative_to(PROJ)),
            "n_chars": n_chars,
            "n_tables": n_tables,
            "status": "too_short",
            "backend": backend,
        }

    md_path.write_text(md, encoding="utf-8")
    return {
        "doi_hash": dh,
        "source_pdf": str(pdf_path.relative_to(PROJ)),
        "n_chars": n_chars,
        "n_tables": n_tables,
        "status": "ok",
        "backend": backend,
    }


# ============================================================
# MinerU backend (optional)
# ============================================================


def parse_one_mineru(pdf_path: Path, output_dir: Path) -> dict:
    """Run mineru CLI on a single PDF. Slower (~30-60s/page) but better tables."""
    dh = _doi_hash(pdf_path)
    md_path = output_dir / f"{dh}.md"
    if md_path.exists() and md_path.stat().st_size > 100:
        return {"doi_hash": dh, "status": "skip_exists", "backend": "cache"}

    tmp_out = output_dir / f"_mineru_tmp_{dh}"
    tmp_out.mkdir(exist_ok=True)
    cmd = [
        "mineru", "-p", str(pdf_path), "-o", str(tmp_out),
        "-b", "pipeline", "-l", "en", "-t", "true", "-f", "false",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode != 0:
            return {"doi_hash": dh, "status": f"mineru_fail: {r.stderr.decode()[:200]}",
                    "backend": "mineru"}
    except subprocess.TimeoutExpired:
        return {"doi_hash": dh, "status": "mineru_timeout", "backend": "mineru"}

    # MinerU writes <pdfname>.md under output dir
    md_candidates = list(tmp_out.glob("**/*.md"))
    if not md_candidates:
        return {"doi_hash": dh, "status": "mineru_no_output", "backend": "mineru"}
    md_text = md_candidates[0].read_text()
    md_path.write_text(md_text, encoding="utf-8")

    # Cleanup tmp
    import shutil
    shutil.rmtree(tmp_out, ignore_errors=True)

    return {
        "doi_hash": dh,
        "source_pdf": str(pdf_path.relative_to(PROJ)),
        "n_chars": len(md_text),
        "n_tables": md_text.count("|---"),  # rough table count
        "status": "ok",
        "backend": "mineru",
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="V56 PDF→Markdown parser")
    ap.add_argument("--backend", choices=["pdfplumber", "mineru"], default="pdfplumber",
                    help="pdfplumber (fast, no model download) or mineru (better tables)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="parse only first N PDFs (smoke test)")
    ap.add_argument("--source", choices=["all", "harvest", "legacy"], default="all")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    if args.source == "all":
        pdfs = _collect_pdfs()
    elif args.source == "harvest":
        pdfs = sorted(HARVEST_DIR.glob("*.pdf")) if HARVEST_DIR.exists() else []
    else:
        pdfs = sorted(LEGACY_DIR.glob("**/*.pdf")) if LEGACY_DIR.exists() else []
    if args.limit:
        pdfs = pdfs[:args.limit]
    logger.info(f"Found {len(pdfs)} PDFs in source={args.source}")

    is_new_manifest = not MANIFEST_CSV.exists()
    fields = ["doi_hash", "source_pdf", "n_chars", "n_tables", "status", "backend"]
    fout = open(MANIFEST_CSV, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if is_new_manifest:
        w.writeheader()

    t0 = time.time()
    n_ok = 0
    n_fail = 0
    n_skip = 0

    if args.backend == "pdfplumber":
        # Parallel ProcessPool — pdfplumber is CPU-bound
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(parse_one_pdfplumber, str(p), str(OUTPUT_DIR)): p for p in pdfs}
            for fut in as_completed(futures):
                row = fut.result()
                w.writerow(row)
                fout.flush()
                status = row.get("status", "?")
                if status == "ok":
                    n_ok += 1
                elif status == "skip_exists":
                    n_skip += 1
                else:
                    n_fail += 1
                if (n_ok + n_fail + n_skip) % 25 == 0:
                    logger.info(f"  Progress: {n_ok} ok, {n_fail} fail, {n_skip} skip "
                                f"(elapsed {(time.time()-t0)/60:.1f}m)")
    else:
        # MinerU sequential — heavy single-PDF resource use
        for p in pdfs:
            row = parse_one_mineru(p, OUTPUT_DIR)
            row.setdefault("source_pdf", str(p.relative_to(PROJ)))
            w.writerow(row)
            fout.flush()
            status = row.get("status", "?")
            if status == "ok":
                n_ok += 1
            elif status == "skip_exists":
                n_skip += 1
            else:
                n_fail += 1
            if (n_ok + n_fail + n_skip) % 5 == 0:
                logger.info(f"  Progress (mineru): {n_ok} ok, {n_fail} fail, {n_skip} skip")

    fout.close()
    logger.info(f"DONE: {n_ok} ok + {n_skip} cached + {n_fail} fail "
                f"(elapsed {(time.time()-t0)/60:.1f}m)")
    logger.info(f"Markdown: {OUTPUT_DIR} | Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
