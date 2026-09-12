"""V56-Ext-2 Stage 0.3 — harvest open-access PDFs for Ga2O3 sputter papers.

Target: ~1,500 PDFs from OpenAlex + Semantic Scholar + arXiv/chemRxiv.

Compliance:
  - OA-only (Phase 55 rule [[feedback_ga2o3_oa_only]])
  - Polite-pool email for OpenAlex/S2 rate-limit privilege
  - Dedup by normalized DOI
  - Workdir for data: PDFs stored in project tree at extra-paper/v56_harvest/

Outputs:
  - extra-paper/v56_harvest/<doi_hash>.pdf       — downloaded OA PDFs
  - extra-paper/v56_harvest/manifest.csv         — (doi, title, year, source, oa_url, pdf_path, status)

Strategy:
  1. OpenAlex /works (primary source — best metadata + OA detection)
     - Query: "Ga2O3" OR "gallium oxide" + (sputter OR photodetector OR doped)
     - year:2018-2026
     - paginate 100/page until ~1000 candidates
  2. Semantic Scholar /paper/search (secondary — different corpus)
     - Query: beta-Ga2O3 thin film dopant photodetector
  3. paperscraper arxiv corpus (tertiary — preprints)

Failures are tolerated (network blips, paywall, 404). Manifest tracks status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import requests

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v56_harvest")


DEFAULT_EMAIL = "haofengli228@gmail.com"
HARVEST_DIR = PROJ / "extra-paper" / "v56_harvest"
MANIFEST_CSV = HARVEST_DIR / "manifest.csv"

HEADERS = {
    "User-Agent": "Ga2O3-Net-Research/0.1 (mailto:%s)" % DEFAULT_EMAIL,
}


def _norm_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    s = str(doi).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def _doi_hash(doi: Optional[str], fallback: str = "") -> str:
    key = doi or fallback
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _load_manifest() -> dict[str, dict]:
    """Load existing manifest if present; key by doi-hash."""
    if not MANIFEST_CSV.exists():
        return {}
    out = {}
    with open(MANIFEST_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["doi_hash"]] = row
    return out


def _append_manifest(rows: list[dict]) -> None:
    """Append rows to manifest CSV, creating header on first write."""
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST_CSV.exists()
    fields = ["doi_hash", "doi", "title", "year", "source", "oa_url",
              "pdf_path", "status", "notes"]
    with open(MANIFEST_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def _download_pdf(url: str, dst: Path, timeout: int = 30) -> bool:
    """Download a PDF from `url` to `dst`. Returns True on success."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return False
        ct = r.headers.get("content-type", "").lower()
        if "pdf" not in ct and not r.content.startswith(b"%PDF"):
            return False
        if len(r.content) < 10000:
            return False  # Likely an error page
        dst.write_bytes(r.content)
        return True
    except Exception as e:
        logger.debug(f"download fail {url}: {e}")
        return False


# ============================================================
# OpenAlex
# ============================================================


def search_openalex(
    query: str,
    year_from: int = 2018,
    year_to: int = 2026,
    per_page: int = 100,
    max_pages: int = 20,
    email: str = DEFAULT_EMAIL,
) -> list[dict]:
    """Search OpenAlex /works. Returns list of work dicts."""
    base = "https://api.openalex.org/works"
    cursor = "*"
    results = []
    for page in range(max_pages):
        params = {
            "search": query,
            "filter": f"publication_year:{year_from}-{year_to},has_doi:true",
            "per_page": per_page,
            "cursor": cursor,
            "mailto": email,
        }
        try:
            r = requests.get(base, params=params, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                logger.warning(f"OpenAlex page {page+1}: HTTP {r.status_code}")
                break
            data = r.json()
        except Exception as e:
            logger.warning(f"OpenAlex page {page+1}: {e}")
            break
        works = data.get("results", [])
        if not works:
            break
        results.extend(works)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        logger.info(f"OpenAlex query='{query[:60]}' page {page+1}: +{len(works)} (total {len(results)})")
        time.sleep(0.2)  # polite pacing
    return results


def harvest_openalex(
    queries: list[str],
    existing_hashes: set[str],
    max_per_query: int = 600,
    email: str = DEFAULT_EMAIL,
) -> tuple[list[dict], list[dict]]:
    """Harvest OpenAlex; download OA PDFs; return (manifest_rows, all_meta)."""
    rows = []
    all_meta = []
    for q in queries:
        works = search_openalex(q, max_pages=max(2, max_per_query // 100), email=email)
        for w in works:
            doi = _norm_doi(w.get("doi"))
            dh = _doi_hash(doi, fallback=w.get("id", ""))
            if dh in existing_hashes:
                continue
            existing_hashes.add(dh)
            all_meta.append(w)

            title = (w.get("title") or "")[:200]
            year = w.get("publication_year")
            oa = w.get("open_access") or {}
            is_oa = bool(oa.get("is_oa"))
            oa_url = oa.get("oa_url")

            pdf_path = HARVEST_DIR / f"{dh}.pdf"
            status = "skipped_paywall"
            if is_oa and oa_url:
                if pdf_path.exists():
                    status = "exists"
                else:
                    ok = _download_pdf(oa_url, pdf_path)
                    status = "downloaded" if ok else "download_failed"

            rows.append({
                "doi_hash": dh,
                "doi": doi or "",
                "title": title,
                "year": year or "",
                "source": "openalex",
                "oa_url": oa_url or "",
                "pdf_path": str(pdf_path.relative_to(PROJ)) if pdf_path.exists() else "",
                "status": status,
                "notes": q[:80],
            })
        _append_manifest(rows[-len(works):])  # incremental save
    return rows, all_meta


# ============================================================
# Semantic Scholar
# ============================================================


def search_semantic_scholar(
    query: str,
    limit: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """Semantic Scholar paper search. Unauthenticated tier: 5000 req/5min."""
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    results = []
    offset = 0
    fields = "title,year,externalIds,openAccessPdf,isOpenAccess,abstract,authors"
    for _ in range(max_pages):
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": fields,
        }
        try:
            r = requests.get(base, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                logger.warning("S2 rate limit; sleeping 30s")
                time.sleep(30)
                continue
            if r.status_code != 200:
                logger.warning(f"S2 offset {offset}: HTTP {r.status_code}")
                break
            data = r.json()
        except Exception as e:
            logger.warning(f"S2 offset {offset}: {e}")
            break
        papers = data.get("data", [])
        if not papers:
            break
        results.extend(papers)
        offset += len(papers)
        if offset >= data.get("total", 0):
            break
        time.sleep(0.4)  # polite pacing
    logger.info(f"S2 query='{query[:60]}': {len(results)} papers")
    return results


def harvest_s2(
    queries: list[str],
    existing_hashes: set[str],
) -> list[dict]:
    rows = []
    for q in queries:
        papers = search_semantic_scholar(q, limit=100, max_pages=3)
        for p in papers:
            ext = p.get("externalIds") or {}
            doi = _norm_doi(ext.get("DOI"))
            dh = _doi_hash(doi, fallback=p.get("paperId", ""))
            if dh in existing_hashes:
                continue
            existing_hashes.add(dh)
            oa_pdf = p.get("openAccessPdf") or {}
            oa_url = oa_pdf.get("url") if oa_pdf else None
            pdf_path = HARVEST_DIR / f"{dh}.pdf"
            status = "skipped_paywall"
            if oa_url:
                if pdf_path.exists():
                    status = "exists"
                else:
                    ok = _download_pdf(oa_url, pdf_path)
                    status = "downloaded" if ok else "download_failed"
            rows.append({
                "doi_hash": dh,
                "doi": doi or "",
                "title": (p.get("title") or "")[:200],
                "year": p.get("year") or "",
                "source": "semantic_scholar",
                "oa_url": oa_url or "",
                "pdf_path": str(pdf_path.relative_to(PROJ)) if pdf_path.exists() else "",
                "status": status,
                "notes": q[:80],
            })
        if rows:
            _append_manifest(rows[-len(papers):])
    return rows


# ============================================================
# Driver
# ============================================================


OPENALEX_QUERIES = [
    "beta-Ga2O3 sputter photodetector",
    "Ga2O3 thin film dopant photoresponse",
    "gallium oxide RF sputtering oxygen vacancy",
    "Ga2O3 doped sputter UV detector",
    "beta gallium oxide solar-blind photodetector",
    "Ga2O3 magnetron sputtering thin film",
]

S2_QUERIES = [
    "beta Ga2O3 thin film dopant photodetector",
    "gallium oxide oxygen vacancy XPS",
    "Ga2O3 sputter solar blind UV detector",
    "Ga2O3 RF magnetron sputtering doping",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="V56 PDF harvest")
    ap.add_argument("--target", type=int, default=1500,
                    help="stop after this many *new* PDFs successfully downloaded")
    ap.add_argument("--skip-openalex", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--email", default=DEFAULT_EMAIL)
    args = ap.parse_args()

    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_manifest()
    existing_hashes = set(existing.keys())
    logger.info(f"Existing manifest: {len(existing_hashes)} entries; target {args.target} new PDFs")

    n_downloaded = sum(1 for r in existing.values() if r.get("status") == "downloaded")

    # Pass 1: OpenAlex
    if not args.skip_openalex:
        logger.info("=== Pass 1: OpenAlex ===")
        rows1, _ = harvest_openalex(OPENALEX_QUERIES, existing_hashes, email=args.email)
        n_new = sum(1 for r in rows1 if r["status"] == "downloaded")
        n_downloaded += n_new
        logger.info(f"OpenAlex: {len(rows1)} candidates, {n_new} new downloads (cumulative {n_downloaded})")

    if n_downloaded >= args.target:
        logger.info(f"Target met after OpenAlex; total downloaded {n_downloaded}")
        return

    # Pass 2: Semantic Scholar
    if not args.skip_s2:
        logger.info("=== Pass 2: Semantic Scholar ===")
        rows2 = harvest_s2(S2_QUERIES, existing_hashes)
        n_new2 = sum(1 for r in rows2 if r["status"] == "downloaded")
        n_downloaded += n_new2
        logger.info(f"S2: {len(rows2)} candidates, {n_new2} new downloads (cumulative {n_downloaded})")

    logger.info(f"=== Harvest done: {n_downloaded} new PDFs downloaded ===")
    logger.info(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
