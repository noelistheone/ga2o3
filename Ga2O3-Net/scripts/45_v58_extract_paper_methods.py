"""
Phase-58 Stage-2: extract Methods/Experimental/Device-fabrication/Measurement
sections from the 659 V56 paper markdowns.

Design-doc reference: §6.3 step 2 ("Section extraction: regex + heuristic
抽取 Methods / Experimental / Experimental section / Device fabrication /
Fabrication of photodetector / Measurement / Optical characterization, 拼接
为单一 methods_section.txt").

Marker-converted PDFs in `extra-paper/v56_markdown/*.md` mostly do NOT carry
proper markdown headings — sections appear as inline numbered headers such
as "2. Experimental" or "2.1 Materials and CuO thin films deposition", or as
bold/uppercase one-line headings.  We therefore use a two-pass scan:

  1. Locate candidate section *start* positions by case-insensitive matching
     against the family of section keywords on a per-line basis (Methods,
     Experimental, Materials and Methods, Device fabrication, Fabrication
     of <X>, Sample preparation, Deposition, Synthesis, Measurement,
     Optical characterization, Characterization, Sputtering procedure,
     Film growth, etc.).  Each candidate is required to look "heading-like":
     short (<= 12 words), starts with an optional "<digit>." or "<digit>.<digit>"
     prefix, and is not embedded mid-sentence.

  2. For each candidate, treat the section body as the text between the
     candidate line and the next *different* top-level numbered heading
     (e.g. "3. Results", "4. Conclusion") OR another heading-like Methods
     keyword OR an "abstract"/"references"/"acknowledgements" line, OR EOF.
     Bodies are deduped and concatenated.

Output:
  - one .txt per paper at `data/processed/paper_methods/<paper_hash>.txt`
  - manifest JSON at `data/processed/paper_methods_manifest.json` with
    {paper_id, source_md, method_text_len, n_sections_matched,
     headings_found, has_methods}.

Skip rules (per task spec):
  - paper not skipped if any heading match; if text < 200 chars or zero
    matches, manifest gets has_methods=False (still written to manifest, but
    no .txt file dropped).

Pure CPU, ~5 minutes for 659 papers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_IN_DIR = HERE / "extra-paper" / "v56_markdown"
DEFAULT_OUT_DIR = HERE / "data" / "processed" / "paper_methods"
DEFAULT_MANIFEST = HERE / "data" / "processed" / "paper_methods_manifest.json"

# Heading-keyword families.  Order does NOT imply priority; ALL matches are
# concatenated.  Lowercase comparison.
# Each entry is a regex matched against a *line* (we strip leading numbering
# like "2.", "2.1", "II.", "(a)").
SECTION_KEYWORD_PATTERNS = [
    # Methods / Experimental / Materials and methods
    r"^(materials\s+and\s+methods?)\b",
    r"^(experimental(\s+section|\s+details|\s+procedure|\s+methods?|\s+setup|\s+part)?)\b",
    r"^(methods?(\s+and\s+materials)?)\b",
    # Device fabrication
    r"^(device\s+fabrication)\b",
    r"^(fabrication\s+(of|and|process|procedure|details|methods?)\b.*)$",
    r"^(fabrication)\b\s*$",
    # Sample preparation / Synthesis / Growth / Deposition
    r"^(sample\s+preparation)\b",
    r"^(film\s+(preparation|growth|deposition|fabrication))\b",
    r"^(thin\s+film\s+(deposition|growth|preparation|fabrication))\b",
    r"^(synthesis(\s+of\s+.*)?)\b",
    r"^(deposition(\s+of\s+.*)?)\b",
    r"^((rf\s+|dc\s+)?(magnetron\s+)?sputter(ing)?(\s+procedure)?)\b",
    r"^(mocvd|cvd|pld|ald|hvpe)\b",
    r"^(growth\s+(of|process|procedure)\b.*)$",
    # Measurement / characterization
    r"^(measurement(s)?(\s+(setup|details|procedure|methods?))?)\b",
    r"^(optical\s+characterization)\b",
    r"^(photo(electric|response|detector|current)\s+measurement(s)?)\b",
    r"^(electrical\s+(measurement|characterization))\b",
    r"^(uv\s+photodetector\s+(measurement|characterization))\b",
    r"^(device\s+characterization)\b",
    r"^(characterization(\s+methods?|\s+techniques?)?)\b\s*$",
]

# Compile (case-insensitive)
_KW_REGEXES = [re.compile(p, re.IGNORECASE) for p in SECTION_KEYWORD_PATTERNS]

# Numbering prefix on a heading line.  e.g. "2. Experimental", "2.1 Materials
# and methods", "II. Methods".
_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|"          # 2.  / 2.1  / 2.1.3
    r"[IVX]+\.\s+|"                          # II.  IV.
    r"\([a-z0-9]\)\s+|"                      # (a)  (1)
    r"#+\s+|"                                # markdown header
    r"\*\*\s*)",                             # bold marker
    re.IGNORECASE,
)

# Headings that *end* the methods region: results / conclusion / references / etc.
END_PATTERNS = [
    r"^(results?(\s+and\s+discussion(s)?)?)\b",
    r"^(discussion(s)?)\b",
    r"^(conclusion(s)?)\b",
    r"^(summary)\b",
    r"^(acknowledg(e)?ments?)\b",
    r"^(references?)\b",
    r"^(bibliography)\b",
    r"^(appendix)\b",
    r"^(supporting\s+information)\b",
    r"^(supplementary)\b",
    r"^(abstract)\b",            # abstract appears BEFORE methods; stop if hit
    r"^(introduction)\b",        # safe: introduction is BEFORE methods
]
_END_REGEXES = [re.compile(p, re.IGNORECASE) for p in END_PATTERNS]


def _strip_numbering(line: str) -> str:
    """Strip leading numbering / bold markup, return the bare heading text."""
    return _NUM_PREFIX_RE.sub("", line).strip()


def _is_heading_like(line: str) -> bool:
    """Heuristic: heading lines are short (<= 12 words) and not a full sentence
    (no terminal period followed by other words after the heading kw)."""
    bare = _strip_numbering(line).strip().rstrip(":.")
    words = bare.split()
    if len(words) == 0 or len(words) > 12:
        return False
    # reject lines with too many sentence-ending tokens
    if bare.count(",") > 2:
        return False
    return True


def _match_kw(line: str) -> str | None:
    """Return the matched heading keyword (lower-cased canonical name) or None."""
    if not _is_heading_like(line):
        return None
    bare = _strip_numbering(line)
    for rx in _KW_REGEXES:
        m = rx.match(bare)
        if m:
            return m.group(1).lower()
    return None


def _match_end(line: str) -> bool:
    if not _is_heading_like(line):
        return False
    bare = _strip_numbering(line)
    for rx in _END_REGEXES:
        if rx.match(bare):
            return True
    return False


def extract_methods_text(md_text: str) -> tuple[str, list[str]]:
    """Scan a paper markdown and return (concatenated_methods, headings_found).

    Strategy:
      - walk lines, when we hit a Methods-family heading start collecting
      - stop collecting when we hit an END heading OR EOF
      - allow multiple Methods sections to be concatenated
    """
    lines = md_text.splitlines()
    out_chunks: list[str] = []
    headings_found: list[str] = []

    in_section = False
    current_chunk: list[str] = []
    current_heading: str | None = None

    def _flush():
        nonlocal current_chunk, current_heading
        if current_chunk and current_heading is not None:
            chunk_text = "\n".join(current_chunk).strip()
            if len(chunk_text) >= 40:  # filter out empty/stub sections
                out_chunks.append(f"## {current_heading}\n{chunk_text}")
                headings_found.append(current_heading)
        current_chunk = []
        current_heading = None

    for raw_line in lines:
        line = raw_line.rstrip()
        kw = _match_kw(line)
        is_end = _match_end(line)

        if kw is not None:
            # start a new section (flush any pending one)
            _flush()
            in_section = True
            current_heading = kw
            continue

        if is_end and in_section:
            _flush()
            in_section = False
            continue

        if in_section:
            current_chunk.append(raw_line)

    _flush()
    return "\n\n".join(out_chunks), headings_found


def process_one(md_path: Path, out_dir: Path) -> dict:
    paper_id = md_path.stem
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {
            "paper_id": paper_id,
            "source_md": str(md_path.relative_to(HERE)),
            "method_text_len": 0,
            "n_sections_matched": 0,
            "headings_found": [],
            "has_methods": False,
            "skip_reason": f"read_error: {e}",
        }

    methods_text, headings = extract_methods_text(md_text)
    n_chars = len(methods_text)

    if n_chars < 200 or not headings:
        return {
            "paper_id": paper_id,
            "source_md": str(md_path.relative_to(HERE)),
            "method_text_len": n_chars,
            "n_sections_matched": len(headings),
            "headings_found": headings,
            "has_methods": False,
            "skip_reason": "too_short" if n_chars < 200 else "no_heading",
        }

    out_path = out_dir / f"{paper_id}.txt"
    out_path.write_text(methods_text, encoding="utf-8")
    return {
        "paper_id": paper_id,
        "source_md": str(md_path.relative_to(HERE)),
        "methods_text_path": str(out_path.relative_to(HERE)),
        "method_text_len": n_chars,
        "n_sections_matched": len(headings),
        "headings_found": headings,
        "has_methods": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN_DIR),
                    help="directory of paper markdowns")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="where to drop per-paper .txt files")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="manifest JSON path")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    md_paths = sorted(in_dir.glob("*.md"))
    print(f"[45_extract_methods] scanning {len(md_paths)} markdowns in {in_dir}")

    records: list[dict] = []
    t0 = time.time()
    for i, md in enumerate(md_paths, 1):
        rec = process_one(md, out_dir)
        records.append(rec)
        if i % 100 == 0 or i == len(md_paths):
            elapsed = time.time() - t0
            n_ok = sum(1 for r in records if r["has_methods"])
            print(f"  [{i}/{len(md_paths)}] has_methods={n_ok}  elapsed={elapsed:.1f}s")

    n_ok = sum(1 for r in records if r["has_methods"])
    n_skip = len(records) - n_ok
    summary = {
        "in_dir": str(in_dir.relative_to(HERE)),
        "out_dir": str(out_dir.relative_to(HERE)),
        "n_papers_scanned": len(records),
        "n_papers_with_methods": n_ok,
        "n_papers_skipped": n_skip,
        "skip_reasons": {
            reason: sum(1 for r in records if r.get("skip_reason") == reason)
            for reason in ("too_short", "no_heading", "read_error")
        },
        "mean_methods_len_chars": (
            float(sum(r["method_text_len"] for r in records if r["has_methods"]))
            / max(1, n_ok)
        ),
        "records": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[45_extract_methods] done.")
    print(f"  with methods: {n_ok}/{len(records)}")
    print(f"  skipped     : {n_skip}  (reasons: {summary['skip_reasons']})")
    print(f"  mean len    : {summary['mean_methods_len_chars']:.0f} chars")
    print(f"  manifest    : {manifest_path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
