"""Phase 60 expansion — best-effort external DFT-vacancy ingestion (Tier-1 data, design §2.5).

Pulls JARVIS `vacancydb` (NIST, DFT vacancy formation energies) via jarvis-tools, filters to
OXIDE / O-vacancy / Ga-containing entries, and saves a compact reusable cache to the workspace
for future encoder pretraining. Honest scope: this enriches DFT grounding / OOD-dopant
generalization; it does NOT address the binding EXPERIMENTAL magnitude ceiling (14 sputter V_O).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data/processed/jarvis_vacancydb_oxides.json"


def main():
    try:
        from jarvis.db.figshare import data
    except Exception as e:
        print("jarvis-tools unavailable:", e); return
    print("downloading JARVIS vacancydb (may take a few min)...", flush=True)
    try:
        d = data("vacancydb")
    except Exception as e:
        print("vacancydb fetch failed:", e); return
    print(f"vacancydb entries: {len(d)}")
    if d:
        print("sample keys:", list(d[0].keys()))
    # JARVIS vacancydb schema: bulk_formula, symbol (vacancy element), ef (formation energy), material_type
    def formula(r):
        return str(r.get("bulk_formula", "")).lower()
    o_vac = [r for r in d if str(r.get("symbol", "")).strip() == "O"]          # oxygen vacancies
    oxide = [r for r in d if "o" in formula(r)]
    ga = [r for r in d if "ga" in formula(r)]
    print(f"O-vacancies: {len(o_vac)}  oxide(formula has O): {len(oxide)}  Ga-containing: {len(ga)}")
    keep_fields = ["jid", "bulk_formula", "symbol", "ef", "material_type", "wycoff"]
    compact = [{k: r.get(k) for k in keep_fields} for r in o_vac]
    OUT.write_text(json.dumps(dict(n_total=len(d), n_O_vacancies=len(o_vac), n_oxide=len(oxide),
                                   n_ga=len(ga), keep_fields=keep_fields, records=compact), default=str))
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(compact)} compact records)")


if __name__ == "__main__":
    main()
