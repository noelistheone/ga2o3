"""R7 reviewer #1 item 7: per-row qualitative disposition of the dark-current and PDR audit rows
that did not value-match (quote_audit_alltables.py, seed 42). Dumps every sampled row with its
recovered quotes and the numeric verdict so each unmatched row can be adjudicated into:
unit-conversion-beyond-registered / range-vs-point / field-association / figure-sourced (no
numbers in text quotes) / genuine mismatch (-> flag). Writes
results/tier2/quote_audit_darkpdr_rows.json."""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qa", PROJ / "scripts/quote_audit_alltables.py.lib" if False else PROJ / "scripts/_qa_lib.py")
# reuse the alltables module by executing it up to its helpers, without rerunning the sweep:
src = (PROJ / "scripts/quote_audit_alltables.py").read_text()
head = src.split("FOMS = [")[0]
ns = {"__file__": str(PROJ / "scripts/quote_audit_alltables.py")}
exec(head, ns)
c72, QUOTES, norm_doi, value_hit, numbers_in = ns["c72"], ns["QUOTES"], ns["norm_doi"], ns["value_hit"], ns["numbers_in"]

out = {}
for fom in ("dark_current_pA", "photo_dark_ratio"):
    d = c72.load_fom(fom)
    sub = d["sub"].reset_index(drop=True)
    sub["_doi"] = [norm_doi(g) for g in np.asarray(d["groups"])]
    doi_ok = sub["_doi"].astype(str).str.startswith(("10.", "arxiv"))
    pool = sub[doi_ok & pd.to_numeric(sub["y"], errors="coerce").notna()]
    samp = pool.sample(n=min(100, len(pool)), random_state=42)
    rows = []
    for idx, r in samp.iterrows():
        qs = QUOTES.get(r["_doi"], [])
        if not qs:
            continue  # LEGACY_NO_QUOTES: outside this disposition (pre-quote era)
        verdict = value_hit(r["y"], qs)
        rows.append({"row_index": int(idx), "doi": r["_doi"], "y": float(r["y"]),
                     "y_lin": float(10 ** r["y"]) if abs(r["y"]) < 25 else None,
                     "verdict": verdict,
                     "quotes": qs if verdict == "absent" else [q[:120] for q in qs[:2]]})
    out[fom] = rows
    n_abs = sum(1 for x in rows if x["verdict"] == "absent")
    print(fom, "joinable", len(rows), "unmatched", n_abs, flush=True)

(PROJ / "results/tier2/quote_audit_darkpdr_rows.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/quote_audit_darkpdr_rows.json")
