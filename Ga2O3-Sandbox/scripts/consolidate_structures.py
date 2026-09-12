"""Consolidate MACE-relaxed doped structures into a per-dopant structure/energy database.

Outputs dft/mace_relax/structure_db.json: per dopant, the relaxed lattice, dopant-O nearest-
neighbour bond lengths, local distortion (spread), energy/atom, and Shannon-radius context.
Also computes a MACE substitution energy proxy vs the undoped reference (per-atom energy delta).
"""
import glob
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
REL = PROJ / "dft/mace_relax/relaxed"

files = sorted(glob.glob(str(REL / "*.json")))
data = {}
for f in files:
    d = json.loads(Path(f).read_text())
    data[d["name"]] = d

undoped = data.get("undoped")
e_und_per_atom = undoped["energy_per_atom"] if undoped else None

db = {"n_structures": len(data), "undoped_energy_per_atom": e_und_per_atom, "dopants": {}}
for name, d in sorted(data.items()):
    if d["dopant"] == "undoped":
        continue
    nn = d.get("dopant_nn_bond_lengths") or []
    nn = [x for x in nn if x < 2.6]     # first coordination shell (M-O bonds)
    entry = {
        "concentration": d["concentration"],
        "n_atoms": d["n_atoms"],
        "energy_per_atom": round(d["energy_per_atom"], 4),
        "dopant_O_bonds_A": [round(x, 3) for x in nn],
        "mean_dopant_O_bond_A": round(float(np.mean(nn)), 3) if nn else None,
        "bond_spread_A": round(float(np.max(nn) - np.min(nn)), 3) if len(nn) > 1 else None,
        "lattice_abc": d.get("lattice_abc"),
        "max_relax_disp_A": round(d.get("max_disp_during_relax", 0), 3),
        "in_KROGER_19": d["dopant"] in ["Si", "H", "Fe", "Sn", "Cr", "Ti", "Ir", "Mg", "Ca",
                                        "Zn", "Co", "Zr", "Hf", "Ta", "Ge", "Pt", "Rh"],
    }
    if e_und_per_atom is not None:
        entry["E_sub_proxy_per_atom_eV"] = round(d["energy_per_atom"] - e_und_per_atom, 4)
    db["dopants"][d["dopant"]] = entry

(PROJ / "dft/mace_relax/structure_db.json").write_text(json.dumps(db, indent=2))

print(f"{len(db['dopants'])} dopant structures consolidated (undoped E/atom={e_und_per_atom})")
print(f"{'dopant':6s} {'inKROGER':9s} {'mean M-O':>9s} {'spread':>7s} {'E/atom':>8s}")
for dop, e in sorted(db["dopants"].items(), key=lambda kv: kv[1]["mean_dopant_O_bond_A"] or 9):
    print(f"{dop:6s} {str(e['in_KROGER_19']):9s} {str(e['mean_dopant_O_bond_A']):>9s} "
          f"{str(e['bond_spread_A']):>7s} {e['energy_per_atom']:>8.3f}")
