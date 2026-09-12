"""
Phase-58 Stage-2: assign 12 binary multi-label protocol classes to each paper
whose Methods section was extracted by `scripts/45_v58_extract_paper_methods.py`.

Design-doc reference: §5.2.3 + §6.3 step 3.  The 12 binary labels are:

    1. target_purity_4N+         (4N / 99.99% / 99.999% mention)
    2. Ar_O2_ratio_oxidizing     (Ar:O2 ratio with non-trivial O2 fraction)
    3. RF_power_high             (RF power >= 100 W mention)
    4. substrate_sapphire        (sapphire / Al2O3 substrate)
    5. anneal_atm_O2             (post-anneal in O2 / oxygen)
    6. anneal_atm_N2             (post-anneal in N2 / nitrogen)
    7. contact_metal_TiAu        (Ti/Au or Au/Ti contact metallurgy)
    8. contact_metal_other       (Pt / Ni / Cr / Ag / In / ITO contact)
    9. illum_254nm               (254 nm illumination — solar-blind regime)
   10. illum_365nm               (365 nm illumination — near-UV / Hg-lamp)
   11. bias_self_powered         ("self-powered" / "0 V bias" / "zero bias")
   12. bias_>=5V                 (5V / 10V / >=5 V bias mention)

Rules are simple keyword + regex on the methods text (case-insensitive).
Each label is binary (0/1); we also keep `n_positive_labels` so the trainer
can downweight all-zero rows if needed (defensive — only logged in manifest).

Output: `data/processed/paper_protocol_labels.jsonl`, one JSON line per
paper with {paper_id, methods_text_path, labels: dict[name->0/1],
n_positive_labels}, plus a small summary JSON sibling.

Pure CPU, runs in seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = HERE / "data" / "processed" / "paper_methods_manifest.json"
DEFAULT_OUT = HERE / "data" / "processed" / "paper_protocol_labels.jsonl"
DEFAULT_SUMMARY = HERE / "data" / "processed" / "paper_protocol_labels_summary.json"


# 12 label NAMES in canonical order (matches ProcessLLM protocol_head 12 logits)
LABEL_NAMES = [
    "target_purity_4N+",
    "Ar_O2_ratio_oxidizing",
    "RF_power_high",
    "substrate_sapphire",
    "anneal_atm_O2",
    "anneal_atm_N2",
    "contact_metal_TiAu",
    "contact_metal_other",
    "illum_254nm",
    "illum_365nm",
    "bias_self_powered",
    "bias_>=5V",
]

# ── Regex rules (case-insensitive on the full methods text) ──────────────────

# 1. 4N+ purity:  "4N" / "99.99%" / "99.999%" / "five 9s"
RX_PURITY_4N = re.compile(
    r"(4n[\s+]|99\.99\s*%|99\.999\s*%|99\.9999\s*%|five\s+nine|5n[\s+]|6n[\s+])",
    re.IGNORECASE,
)

# 2. Ar:O2 ratio with non-trivial O2 (any X:Y where Y>=1, OR an explicit
#    O2 fraction in sccm/percent that suggests an oxidizing or mixed atm).
#    We pattern-match Ar:O2 = N:M with M >= 1, or O2 ≥ 10% / O2 sccm ≥ 1.
RX_AR_O2_RATIO = re.compile(
    r"(?:ar\s*:\s*o\s*[_2]?\s*=?\s*\d+\s*:\s*[1-9]\d*"
    r"|ar/o\s*[_2]?\s*=?\s*\d+\s*/\s*[1-9]\d*"
    r"|o[\s_-]*2\s*(?:flow|content|fraction)\s*(?:of)?\s*(?:[1-9]\d?)\s*%"
    r"|o[\s_-]*2\s*(?:partial\s+pressure|flow\s+rate)\s*(?:of)?\s*[1-9])",
    re.IGNORECASE,
)
# Also fire if anything like "in O2" / "oxygen atmosphere" appears alongside Ar
RX_OXIDIZING_ATMO = re.compile(
    r"(oxygen\s+(atmosphere|ambient|gas)"
    r"|o\s*[_2]?\s+(atmosphere|ambient|partial\s+pressure)"
    r"|ar\s*[/+]?\s*o\s*[_2]?\s+(mixture|atmosphere|plasma))",
    re.IGNORECASE,
)

# 3. RF power high (>=100 W) — capture "RF power of 120 W" / "RF: 200W" etc.
RX_RF_POWER = re.compile(
    r"(?:rf\s+power|sputter(?:ing)?\s+power|power\s+of)\s*(?:of|=|:|was|is)?\s*"
    r"(\d{2,4}(?:\.\d+)?)\s*w\b",
    re.IGNORECASE,
)

# 4. sapphire substrate
RX_SUBSTRATE_SAPPHIRE = re.compile(
    r"(sapphire|al\s*[_2]?o\s*[_3]?\s+substrate|c-?plane\s+sapphire|c-?cut\s+sapphire)",
    re.IGNORECASE,
)

# 5. anneal in O2
RX_ANNEAL_O2 = re.compile(
    r"(?:anneal(?:ed|ing)?|post[-\s]?annealing|rta)[^.]{0,80}?"
    r"(?:in|under|with)\s+(?:flowing\s+)?o(?:xygen|\s*[_2]?)\b",
    re.IGNORECASE,
)

# 6. anneal in N2
RX_ANNEAL_N2 = re.compile(
    r"(?:anneal(?:ed|ing)?|post[-\s]?annealing|rta)[^.]{0,80}?"
    r"(?:in|under|with)\s+(?:flowing\s+)?n(?:itrogen|\s*[_2]?)\b",
    re.IGNORECASE,
)

# 7. Ti/Au or Au/Ti contact
RX_CONTACT_TIAU = re.compile(
    r"(?:ti\s*/\s*au|au\s*/\s*ti|titanium\s*/\s*gold|gold\s*/\s*titanium)",
    re.IGNORECASE,
)

# 8. other contact metal — require explicit electrode/contact context to avoid
#    false positives from element names appearing in other compounds (e.g. "Pt
#    nanoparticles" / "Cu-doped").  We require the metal name within 30 chars
#    of "electrode", "contact", "ohmic", "schottky", or "interdigital".
RX_CONTACT_OTHER = re.compile(
    r"(?:(?:\bpt\b|\bplatinum\b|\bni\b|\bnickel\b|\bcr\b|\bchromium\b|"
    r"\bag\b|\bsilver\b|\bito\b|\bcu\b|\bal\b|\bw\b|\bmo\b)\W{0,30}"
    r"(?:electrode|contact|ohmic|schottky|interdigit|finger)"
    r"|(?:electrode|contact|ohmic|schottky|interdigit|finger)\W{0,30}"
    r"(?:\bpt\b|\bplatinum\b|\bni\b|\bnickel\b|\bcr\b|\bchromium\b|"
    r"\bag\b|\bsilver\b|\bito\b|\bcu\b|\bal\b|\bw\b|\bmo\b))",
    re.IGNORECASE,
)

# 9. 254 nm illumination
RX_ILLUM_254 = re.compile(r"254\s*nm", re.IGNORECASE)

# 10. 365 nm illumination
RX_ILLUM_365 = re.compile(r"365\s*nm", re.IGNORECASE)

# 11. self-powered / zero-bias
RX_BIAS_SELF = re.compile(
    r"(self[-\s]?powered|zero\s*[-\s]?bias|0\s*v\s*bias|self[-\s]?driven)",
    re.IGNORECASE,
)

# 12. bias >= 5V
RX_BIAS_HIGH = re.compile(
    r"(?:bias|voltage)\s+(?:of\s+)?(\d{1,3}(?:\.\d+)?)\s*v\b",
    re.IGNORECASE,
)


def _rf_power_high(text: str) -> int:
    for m in RX_RF_POWER.finditer(text):
        try:
            if float(m.group(1)) >= 100.0:
                return 1
        except ValueError:
            continue
    return 0


def _bias_high(text: str) -> int:
    for m in RX_BIAS_HIGH.finditer(text):
        try:
            if float(m.group(1)) >= 5.0:
                return 1
        except ValueError:
            continue
    return 0


def assign_labels(text: str) -> dict[str, int]:
    """Return the 12-class binary label dict for one methods-text string."""
    labels = {name: 0 for name in LABEL_NAMES}
    if not text:
        return labels

    labels["target_purity_4N+"] = int(bool(RX_PURITY_4N.search(text)))
    labels["Ar_O2_ratio_oxidizing"] = int(
        bool(RX_AR_O2_RATIO.search(text)) or bool(RX_OXIDIZING_ATMO.search(text))
    )
    labels["RF_power_high"] = _rf_power_high(text)
    labels["substrate_sapphire"] = int(bool(RX_SUBSTRATE_SAPPHIRE.search(text)))
    labels["anneal_atm_O2"] = int(bool(RX_ANNEAL_O2.search(text)))
    labels["anneal_atm_N2"] = int(bool(RX_ANNEAL_N2.search(text)))
    labels["contact_metal_TiAu"] = int(bool(RX_CONTACT_TIAU.search(text)))
    labels["contact_metal_other"] = int(bool(RX_CONTACT_OTHER.search(text)))
    labels["illum_254nm"] = int(bool(RX_ILLUM_254.search(text)))
    labels["illum_365nm"] = int(bool(RX_ILLUM_365.search(text)))
    labels["bias_self_powered"] = int(bool(RX_BIAS_SELF.search(text)))
    labels["bias_>=5V"] = _bias_high(text)
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n_total = 0
    n_written = 0
    label_counts = {name: 0 for name in LABEL_NAMES}
    pos_per_paper: list[int] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in manifest["records"]:
            if not rec.get("has_methods"):
                continue
            n_total += 1
            txt_rel = rec["methods_text_path"]
            txt_abs = (HERE / txt_rel).resolve()
            try:
                text = txt_abs.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                continue

            labels = assign_labels(text)
            n_pos = sum(labels.values())
            for k, v in labels.items():
                label_counts[k] += int(v)
            pos_per_paper.append(n_pos)

            row = {
                "paper_id": rec["paper_id"],
                "methods_text_path": txt_rel,
                "method_text_len": rec["method_text_len"],
                "labels": labels,
                "n_positive_labels": n_pos,
            }
            fout.write(json.dumps(row) + "\n")
            n_written += 1

    summary = {
        "manifest_used": args.manifest,
        "n_papers_with_methods": n_total,
        "n_papers_labeled": n_written,
        "label_positive_counts": label_counts,
        "label_positive_rates": {
            k: (v / max(1, n_written)) for k, v in label_counts.items()
        },
        "mean_n_positive_per_paper": (
            float(sum(pos_per_paper)) / max(1, len(pos_per_paper))
        ),
        "n_papers_with_zero_positives": sum(1 for n in pos_per_paper if n == 0),
        "n_papers_with_>=3_positives": sum(1 for n in pos_per_paper if n >= 3),
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[46_assign_protocol_labels] done.")
    print(f"  papers labeled              : {n_written} / {n_total}")
    print(f"  mean positives per paper    : {summary['mean_n_positive_per_paper']:.2f}")
    print(f"  zero-positive papers        : {summary['n_papers_with_zero_positives']}")
    print(f"  >=3-positive papers         : {summary['n_papers_with_>=3_positives']}")
    print(f"  label positive rates        :")
    for k in LABEL_NAMES:
        rate = summary["label_positive_rates"][k]
        print(f"    {k:<24} : {label_counts[k]:>4}  ({rate*100:5.1f}%)")
    print(f"  output: {out_path}")
    print(f"  summary: {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
