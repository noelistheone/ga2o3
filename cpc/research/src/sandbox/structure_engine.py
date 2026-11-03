"""structure_engine — the objective-accuracy (MP-pretrained) structure layer.

Harvested from Ga2O3-Net's MACE-MPA-0 (Tier 3, 2026-07-07). This is the sandbox's
STRUCTURE engine: it produces relaxed doped / defect geometries with DFT-quality accuracy
that is INDEPENDENT of the literature-lab confound (trained on Materials-Project DFT, not
on the corpus). It deliberately does NOT produce energetics — those go through QE/HSE
(Tier B), because the benchmark showed MLIP energetics are unusable while MLIP geometry is
excellent.

Validated bounds (docs/tier3_mlip_energetics.md, results/tier3/*):
  - geometry: MACE-MPA-0 relaxed defect geom vs QE-relaxed  RMSD 0.014-0.054 A (5/6 gate PASS)
  - the ONE failure: negative-U V_O(2+) (RMSD 0.165 A) — MACE is charge-blind and misses the
    large charge-driven relaxation. Rule below: flag large-ΔQ charged states for HSE relax.
  - energy penalty of using the MACE geometry instead of a QE relax: +0.155 eV for V_O
    -> protocol is MACE pre-relax then a SHORT QE/HSE final relax (not a bare single-point)
    for absolute energetics; MACE geometry alone is fine for structure/screening.
  - zero-shot MLIP formation energy: off +0.9..+1.6 eV (DO NOT use for energetics).

Model: ~/.cache/mace/macempa0mediummodel  (MACE-MPA-0 medium, 9.06M params).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

MODEL_PATH = "/home/lawrence/.cache/mace/macempa0mediummodel"
PERFECT_CELL = Path(__file__).resolve().parents[2] / "dft/qe_cc/gate2_perfect.in"
O_CUT = 2.4          # A, metal-O neighbour cutoff
FMAX = 0.02
STEPS = 500

# Validated accuracy metadata (for provenance / fidelity tagging)
GEOM_RMSD_VS_QE = {"V_O_q0": 0.026, "V_Ga_q0": 0.054, "V_Ga_qm3": 0.044,
                   "Fe_Ga_q0": 0.017, "Fe_Ga_qm1": 0.014, "V_O_q2": 0.165}
GEOM_ENERGY_PENALTY_eV = 0.155      # V_O neutral: E(MACE geom) - E(QE min), PBE
DQ_CHARGED_FLAG = 6.0               # amu^0.5 A: above this, MLIP charged geometry is untrusted


def _calc(device="cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=MODEL_PATH, device=device, default_dtype="float64")


def relax(atoms, cell=False, calc=None, device="cuda"):
    """Relax atoms (optionally cell). Returns (relaxed_atoms, energy, max_force)."""
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    a = atoms.copy()
    a.calc = calc or _calc(device)
    FIRE(FrechetCellFilter(a) if cell else a, logfile=None).run(fmax=FMAX, steps=STEPS)
    return a, float(a.get_potential_energy()), float(np.abs(a.get_forces()).max())


def ga_site_coordination(atoms):
    """Classify Ga sites by O-coordination: 4 -> tetrahedral (I), >=5 -> octahedral (II)."""
    pos = atoms.get_positions(); sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.cell)
    o_idx = np.where(sym == "O")[0]
    out = {}
    for gi in np.where(sym == "Ga")[0]:
        d = pos[o_idx] - pos[gi]
        frac = np.linalg.solve(cell.T, d.T).T; frac -= np.round(frac)
        out[int(gi)] = int((np.linalg.norm(frac @ cell, axis=1) < O_CUT).sum())
    return out


def new_dopant_defect(dopant, site="oct", calc=None, device="cuda"):
    """Build + MACE-relax a substitutional dopant on the chosen Ga site of the 80-atom cell.
    Returns dict with relaxed atoms, dopant-O bonds, coordination, MACE energy.
    site: 'oct' (Ga_II, octahedral) or 'tet' (Ga_I, tetrahedral)."""
    import ase.io
    calc = calc or _calc(device)
    perfect = ase.io.read(str(PERFECT_CELL), format="espresso-in")
    coord = ga_site_coordination(perfect)
    want = 4 if site == "tet" else 6
    cands = [i for i, c in coord.items() if (c == 4) == (want == 4)]
    gi = cands[0]
    a = perfect.copy(); a[gi].symbol = dopant
    a, E, fmax = relax(a, cell=False, calc=calc)
    # dopant-O bonds
    pos = a.get_positions(); sym = np.array(a.get_chemical_symbols()); cell = np.array(a.cell)
    o_idx = np.where(sym == "O")[0]
    d = pos[o_idx] - pos[gi]
    frac = np.linalg.solve(cell.T, d.T).T; frac -= np.round(frac)
    dist = np.sort(np.linalg.norm(frac @ cell, axis=1))
    bonds = dist[dist < O_CUT + 0.4]
    return {"atoms": a, "sub_index": int(gi), "E_mace_eV": E,
            "dop_O_bonds_A": [round(float(b), 3) for b in bonds],
            "mean_MO_A": round(float(bonds.mean()), 3), "coordination": int(len(bonds))}


def geometry_fidelity(mace_atoms, qe_atoms):
    """RMSD between two same-order, same-cell geometries (minimum image). < 0.1 A => MLIP
    geometry can replace the QE relaxation for that defect (Tier B)."""
    cell = np.array(mace_atoms.cell)
    d = np.array(qe_atoms.positions) - np.array(mace_atoms.positions)
    frac = np.linalg.solve(cell.T, d.T).T; frac -= np.round(frac)
    dn = np.linalg.norm(frac @ cell, axis=1)
    return {"rmsd_A": float(np.sqrt((dn ** 2).mean())), "max_A": float(dn.max()),
            "pass_0.1A": bool(np.sqrt((dn ** 2).mean()) < 0.1)}


def needs_charged_dft(defect_dQ):
    """Tier-B gate: MLIP geometry is trustworthy for neutral & weak-coupling charged defects,
    but NOT for negative-U / strongly charge-coupled defects (large ΔQ between charge states,
    e.g. V_O 2+ with ΔQ=12.3). Those need a charged-DFT/HSE final relaxation. The sandbox
    already computes ΔQ per defect in the tau campaign, so this gate is auto-applicable."""
    return bool(defect_dQ is not None and defect_dQ > DQ_CHARGED_FLAG)
