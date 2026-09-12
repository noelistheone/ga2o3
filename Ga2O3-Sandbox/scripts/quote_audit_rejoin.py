"""Re-join the 42 UNJUDGEABLE audit rows (quote field = 'phase77_ingest' provenance tag) to the
ingest-time verification artifacts in Ga2O3-Net (READ-ONLY), which retain per-row verbatim source
quotes, and re-grade each row by numeric containment of its transport values in the recovered
quotes. Writes results/tier2/quote_audit_rejoin.json (verdicts + evidence quotes)."""
import json, re
from pathlib import Path
import pandas as pd
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"

audit = json.load(open(PROJ / "results/tier2/quote_audit_100.json"))
unj = [r for r in audit["per_row"] if r["verdict"] == "UNJUDGEABLE"]
tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")


def norm_doi(d):
    if d is None:
        return None
    d = str(d).strip().lower()
    d = d.split(" (")[0]                       # strip parenthetical mirrors
    m = re.search(r"(?:arxiv\.org/abs/|10\.48550/arxiv\.)(\d{4}\.\d{4,5})", d)
    if m:
        return "arxiv:" + m.group(1)
    return d


# ---- harvest quote stores: recursively collect {doi -> [quotes]} ----
QUOTES = {}


def add(doi, q):
    if doi and q and isinstance(q, str) and len(q) > 10 and q != "phase77_ingest":
        QUOTES.setdefault(norm_doi(doi), []).append(q)


def walk(node, doi=None):
    if isinstance(node, dict):
        doi = node.get("doi", doi)
        if "quote" in node:
            add(doi, node["quote"])
        for v in node.values():
            walk(v, doi)
    elif isinstance(node, list):
        for v in node:
            walk(v, doi)


for f in ("deduped.json", "verify_191_result.json", "deepen_result.json"):
    walk(json.load(open(NET / "results/phase77_new_data_sweep" / f)))
for d, qs in QUOTES.items():
    QUOTES[d] = sorted(set(qs))

# ---- numeric extraction from a quote ----
SCI = re.compile(r"(\d+(?:\.\d+)?)\s*[x×*]\s*10\s*[\^{]*\s*(-?\d+)")
SUP = re.compile(r"(\d+(?:\.\d+)?)\s*[x×*]\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
SUPMAP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
PLAIN = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(q):
    out = []
    for m in SCI.finditer(q):
        out.append(float(m.group(1)) * 10 ** int(m.group(2)))
    for m in SUP.finditer(q):
        out.append(float(m.group(1)) * 10 ** int(m.group(2).translate(SUPMAP)))
    stripped = SCI.sub(" ", SUP.sub(" ", q))
    out += [float(x) for x in PLAIN.findall(stripped)]
    return out


def value_in_quotes(v, quotes):
    """exact = rel-1% match; mantissa = mantissa-only match (exponent lost in text)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    mant = v / 10 ** np.floor(np.log10(abs(v))) if v != 0 else 0.0
    hit = ""
    for q in quotes:
        nums = numbers_in(q)
        if any(abs(x - v) <= 0.01 * abs(v) for x in nums):
            return ("exact", q)
        if not hit and any(abs(x - mant) <= 0.005 * max(abs(mant), 1) for x in nums):
            hit = ("mantissa", q)
    return hit or ("absent", None)


FIELDS = ["carrier_cm3", "mu_cm2Vs", "rho_ohmcm", "concentration_value"]
rows_out, counts = [], {"SUPPORTED": 0, "PARTIAL": 0, "NOT_FOUND": 0, "NO_QUOTES": 0}
for r in unj:
    src = tp.loc[r["index"]]
    doi = norm_doi(src["doi"])
    quotes = QUOTES.get(doi, [])
    checked, found, evid = [], [], {}
    for f in FIELDS:
        v = src.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        checked.append(f)
        if quotes:
            res = value_in_quotes(v, quotes)
            if res[0] in ("exact", "mantissa"):
                found.append(f)
                evid[f] = {"value": float(v), "match": res[0], "quote": res[1][:300]}
    if not quotes:
        verdict = "NO_QUOTES"
    elif len(found) == len(checked) and checked:
        verdict = "SUPPORTED"
    elif found:
        verdict = "PARTIAL"
    else:
        verdict = "NOT_FOUND"
    counts[verdict] += 1
    rows_out.append({"index": r["index"], "doi": src["doi"], "checked": checked,
                     "found": found, "verdict": verdict, "evidence": evid,
                     "n_candidate_quotes": len(quotes)})

out = {
    "n_rejoined": len(unj), "counts": counts,
    "method": ("The 42 audit rows whose in-table quote was the provenance tag 'phase77_ingest' "
               "were re-joined by normalized DOI to the ingest-time verification artifacts "
               "(Ga2O3-Net results/phase77_new_data_sweep/{deduped,verify_191_result,"
               "deepen_result}.json), which retain per-row verbatim source quotes. Each numeric "
               "field of the audited row was then searched in the recovered quotes (scientific-"
               "notation-aware, rel. 1%); SUPPORTED = every non-null field found, PARTIAL = some, "
               "NOT_FOUND = none, NO_QUOTES = DOI absent from the artifacts."),
    "per_row": rows_out,
}
(PROJ / "results/tier2/quote_audit_rejoin.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"counts": counts}, indent=1))
for r in rows_out:
    if r["verdict"] in ("NOT_FOUND", "NO_QUOTES"):
        print(r["index"], r["doi"], r["verdict"], r["checked"], "quotes:", r["n_candidate_quotes"])
