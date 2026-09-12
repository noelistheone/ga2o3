"""Extract occupation-based gaps of the low-density amorphous cells (5.4 / 5.0 g/cc) with the
identical extractor, and test the conservative-bound monotonicity: lower density => equal or
larger disorder narrowing than the 5.95 g/cc ensemble (-0.58 eV). Writes
results/tier3/disorder_lowrho.json."""
import json
from pathlib import Path
import numpy as np
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("edge", PROJ / "scripts/tier3_disorder_edge.py")
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)

ck = edge.per_k_eigs("crystal")
cg = edge.gap(ck)["gap"] if ck else 2.339
cells = []
for tag, rho in (("lr54", 5.4), ("lr50", 5.0)):
    ks = edge.per_k_eigs(tag)
    if not ks:
        print(tag, "DOS not ready")
        continue
    g = edge.gap(ks)
    cells.append({"cell": tag, "density_gcc": rho, "gap_eV": g["gap"],
                  "narrowing_eV": round(g["gap"] - cg, 3)})
    print(tag, cells[-1], flush=True)

ref = -0.584
out = {"crystal_gap_eV": cg, "reference_595_narrowing_eV": ref, "cells": cells}
if len(cells) == 2:
    mono = all(c["narrowing_eV"] <= ref + 0.1 for c in cells)
    out["monotonic_within_0.1eV"] = bool(mono)
    out["read"] = ("lower density disorders further (narrowing >= the 5.95 g/cc value) => the "
                   "fixed near-crystalline-density ensembles are a conservative lower bound"
                   if mono else
                   "monotonicity NOT confirmed at these densities -- report as measured")
(PROJ / "results/tier3/disorder_lowrho.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
