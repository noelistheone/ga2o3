"""Tier-B new-dopant front-end: extend the sandbox beyond KROGER's 19 elements.

For a dopant outside KROGER (Sb, Bi, ...), build the substitutional defect on BOTH
inequivalent Ga sites (tetrahedral Ga_I, octahedral Ga_II) in the 80-atom beta-Ga2O3
cell, MACE-MPA-0 relax each (the validated structure step), and report:
  - site preference (relative MACE energy — a within-composition relative energy MLIPs
    handle well)
  - the relaxed dopant local structure (M-O bonds, coordination) = the "sputter-doped
    atomic structure"
  - writes a QE scf input at the MACE geometry of the preferred site, for a PBE
    single-point now / HSE single-point later (Tier B energetics).

Usage: tier3_new_dopant_tierB.py Sb [Bi ...]
Writes results/tier3/new_dopant_<M>.json and dft/qe_cc/dop_<M>_Ga{I,II}_at_mace.in
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
import ase.io
from ase.optimize import FIRE

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
OUT = PROJ / "results/tier3"
OUT.mkdir(parents=True, exist_ok=True)
FMAX, STEPS = 0.02, 500
O_CUT = 2.4   # Ang, Ga-O neighbor cutoff


def mace_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel",
                   device="cuda", default_dtype="float64")


def ga_sites(atoms):
    """Classify Ga sites by O-coordination: 4 -> tetrahedral (I), 6 -> octahedral (II)."""
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.cell)
    ga_idx = np.where(sym == "Ga")[0]
    o_idx = np.where(sym == "O")[0]
    sites = {}
    for gi in ga_idx:
        d = pos[o_idx] - pos[gi]
        frac = np.linalg.solve(cell.T, d.T).T
        frac -= np.round(frac)
        dist = np.linalg.norm(frac @ cell, axis=1)
        coord = int((dist < O_CUT).sum())
        sites[int(gi)] = coord
    return sites


def dop_bonds(atoms, dop_idx):
    pos = atoms.get_positions(); sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.cell)
    o_idx = np.where(sym == "O")[0]
    d = pos[o_idx] - pos[dop_idx]
    frac = np.linalg.solve(cell.T, d.T).T; frac -= np.round(frac)
    dist = np.sort(np.linalg.norm(frac @ cell, axis=1))
    bonds = dist[dist < O_CUT + 0.4][:6]
    return bonds


def write_qe_scf(atoms, prefix, dop_elem):
    """Write a QE-PBE scf single-point at this geometry (template from cc_VO_q0_relax.in)."""
    src = (CC / "cc_VO_q0_relax.in").read_text()
    t = src
    t = t.replace("calculation = 'relax'", "calculation = 'scf'")
    t = re.sub(r"\n\s*nstep\s*=.*", "", t)
    t = re.sub(r"\n\s*forc_conv_thr\s*=.*", "", t)
    t = re.sub(r"prefix = '[^']+'", f"prefix = '{prefix}'", t)
    t = re.sub(r"outdir = '[^']+'", f"outdir = '{CC / 'outdir' / prefix}'", t)
    t = re.sub(r"&IONS.*?/\n", "", t, flags=re.S)
    t = re.sub(r"nat = \d+", f"nat = {len(atoms)}", t)
    t = re.sub(r"ntyp = \d+", "ntyp = 3", t)
    # add dopant species line (pseudo must be supplied later; placeholder name)
    t = re.sub(r"(ATOMIC_SPECIES\n)", rf"\1  {dop_elem:<4s}  1.0  {dop_elem}.upf\n", t)
    pos = "ATOMIC_POSITIONS angstrom\n"
    for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        pos += f"  {s:<4s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}\n"
    t = re.sub(r"ATOMIC_POSITIONS[^\n]*\n(?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+",
               pos, t)
    (CC / f"{prefix}.in").write_text(t)


def run_dopant(elem, perfect, calc):
    sites = ga_sites(perfect)
    tet = [i for i, c in sites.items() if c == 4]
    oct_ = [i for i, c in sites.items() if c >= 5]
    pick = {"Ga_I_tet": tet[0] if tet else None, "Ga_II_oct": oct_[0] if oct_ else None}
    res = {"dopant": elem, "n_tet_sites": len(tet), "n_oct_sites": len(oct_), "configs": {}}
    energies = {}
    for label, gi in pick.items():
        if gi is None:
            continue
        a = perfect.copy()
        a[gi].symbol = elem
        a.calc = calc
        FIRE(a, logfile=None).run(fmax=FMAX, steps=STEPS)
        E = float(a.get_potential_energy())
        bonds = dop_bonds(a, gi)
        energies[label] = E
        res["configs"][label] = {
            "sub_index": int(gi),
            "E_mace_eV": round(E, 4),
            "dop_O_bonds_A": [round(float(b), 3) for b in bonds],
            "mean_MO_A": round(float(bonds.mean()), 3),
            "coordination": int(len(bonds)),
        }
        ase.io.write(str(CC / f"dop_{elem}_{label}_mace.xyz"), a)
        write_qe_scf(a, f"dop_{elem}_{label}_at_mace", elem)
    if len(energies) == 2:
        pref = min(energies, key=energies.get)
        dE = abs(energies["Ga_I_tet"] - energies["Ga_II_oct"])
        res["preferred_site"] = pref
        res["site_pref_dE_eV"] = round(dE, 3)
    (OUT / f"new_dopant_{elem}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    dopants = sys.argv[1:] or ["Sb", "Bi"]
    perfect = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")
    calc = mace_calc()
    for elem in dopants:
        r = run_dopant(elem, perfect, calc)
        c = r["configs"]
        print(f"[{elem}] preferred={r.get('preferred_site')} dE={r.get('site_pref_dE_eV')} eV", flush=True)
        for lab, cc in c.items():
            print(f"    {lab}: mean {elem}-O = {cc['mean_MO_A']} A, coord {cc['coordination']}, "
                  f"bonds {cc['dop_O_bonds_A']}", flush=True)
    print("WROTE results/tier3/new_dopant_*.json + QE inputs", flush=True)


if __name__ == "__main__":
    main()
