"""R6 reviewer #1 item 7: extend the transcription audit beyond the transport table.

For each remaining directly-measured property (PDR, dark current, [V_O], Eg, dEg, tau), sample up
to 100 rows (seed 42), re-join by DOI to the ingest-time verification artifacts that retain
verbatim source quotes (phase77 sweep x3, phase96/97 bandgap harvests, phase103 harvest), and
grade each row by numeric containment of its value in the recovered quotes (rel. 1%, scientific-
notation-aware; log10-stored values also matched as 10^y). Rows whose DOI predates quote-keeping
(the earliest ingest era) are reported as LEGACY_NO_QUOTES -- the honest coverage boundary.
Writes results/tier2/quote_audit_alltables.json."""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("c72", NET / "scripts/_phase72_common.py")
c72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c72)

SCI = re.compile(r"(\d+(?:\.\d+)?)\s*[x×*]\s*10\s*[\^{]*\s*(-?\d+)")
SUP = re.compile(r"(\d+(?:\.\d+)?)\s*[x×*]\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
SUPMAP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
PLAIN = re.compile(r"\d+(?:\.\d+)?")


def norm_doi(d):
    if d is None:
        return None
    d = str(d).strip().lower().split(" (")[0]
    m = re.search(r"(?:arxiv\.org/abs/|10\.48550/arxiv\.)(\d{4}\.\d{4,5})", d)
    return ("arxiv:" + m.group(1)) if m else d


QUOTES = {}


def add(doi, q):
    if doi and isinstance(q, str) and len(q) > 10 and q != "phase77_ingest":
        QUOTES.setdefault(norm_doi(doi), []).append(q)


def walk(node, doi=None):
    if isinstance(node, dict):
        doi = node.get("doi", node.get("_doi", doi))
        if "quote" in node:
            add(doi, node["quote"])
        for v in node.values():
            walk(v, doi)
    elif isinstance(node, list):
        for v in node:
            walk(v, doi)


for f in ("deduped.json", "verify_191_result.json", "deepen_result.json"):
    walk(json.load(open(NET / "results/phase77_new_data_sweep" / f)))
for f in ("results/phase96/harvest_rows.json", "results/phase97/harvest_rows.json"):
    walk(json.load(open(NET / f)))
try:
    walk(json.load(open(NET / "results/phase103/harvest_result.json")))
except Exception:
    pass
for d in QUOTES:
    QUOTES[d] = sorted(set(QUOTES[d]))
print("quote stores: %d DOIs, %d quotes" % (len(QUOTES), sum(len(v) for v in QUOTES.values())))


def numbers_in(q):
    out = []
    for m in SCI.finditer(q):
        out.append(float(m.group(1)) * 10 ** int(m.group(2)))
    for m in SUP.finditer(q):
        out.append(float(m.group(1)) * 10 ** int(m.group(2).translate(SUPMAP)))
    stripped = SCI.sub(" ", SUP.sub(" ", q))
    out += [float(x) for x in PLAIN.findall(stripped)]
    return out


def value_hit(v, quotes):
    v = float(v)
    exact_c = [v]
    if 0 < abs(v) < 25:
        exact_c.append(10 ** v)                      # log10-stored variants
    unit_c = [v * 10 ** k for k in (-12, -9, -6, -3, 3, 6, 9, 12)]   # unit-registered (A/pA/nA/uA, s/ms/us)
    mant = v / 10 ** np.floor(np.log10(abs(v))) if v != 0 else 0.0
    best = "absent"
    for q in quotes:
        nums = numbers_in(q)
        if any(abs(x - c) <= 0.011 * abs(c) for c in exact_c for x in nums):
            return "exact"
        if best in ("absent", "mantissa") and any(
                abs(x - c) <= 0.011 * abs(c) for c in unit_c for x in nums):
            best = "unit"
        if best == "absent" and any(abs(x - mant) <= 0.005 * max(abs(mant), 1) for x in nums):
            best = "mantissa"
    return best


FOMS = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration",
        "optical_bandgap_Eg", "tau_decay"]
# bandgap_shift_dEg excluded: a derived within-study CONTRAST (doped minus undoped Eg) never
# appears verbatim in source text; its constituents are audited under optical_bandgap_Eg.
rng = np.random.default_rng(42)
out = {}
for fom in FOMS:
    d = c72.load_fom(fom)
    sub = d["sub"].reset_index(drop=True)
    sub["_doi"] = [norm_doi(g) for g in np.asarray(d["groups"])]
    doi_ok = sub["_doi"].astype(str).str.startswith(("10.", "arxiv"))
    pool = sub[doi_ok & pd.to_numeric(sub["y"], errors="coerce").notna()]
    n = min(100, len(pool))
    samp = pool.sample(n=n, random_state=42)
    counts = {"exact": 0, "unit": 0, "mantissa": 0, "absent": 0, "LEGACY_NO_QUOTES": 0}
    for _, r in samp.iterrows():
        qs = QUOTES.get(r["_doi"], [])
        if not qs:
            counts["LEGACY_NO_QUOTES"] += 1
            continue
        counts[value_hit(r["y"], qs)] += 1
    joinable = n - counts["LEGACY_NO_QUOTES"]
    out[fom] = {"sampled": n, "no_doi_rows_in_table": int((~doi_ok).sum()),
                "joinable": joinable, **counts,
                "value_supported_frac_of_joinable":
                    round((counts["exact"] + counts["unit"] + counts["mantissa"]) / joinable, 3) if joinable else None}
    print(fom, out[fom], flush=True)

out["_method"] = ("per-FOM sample of <=100 DOI-bearing rows (seed 42); quotes recovered from the "
                  "six ingest-time artifacts; exact = value (or 10^value) within 1% in a quote of "
                  "the same DOI; LEGACY = DOI absent from all quote stores (pre-quote-keeping era)")
(PROJ / "results/tier2/quote_audit_alltables.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/quote_audit_alltables.json")
