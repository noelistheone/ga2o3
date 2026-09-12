"""Extract the FUNDAMENTAL band gap + Urbach energy from the disorder-ensemble QE-PBE DOS
outputs and compute the crystal->amorphous gap narrowing (disorder-dEg, structural component).

Occupation-based fundamental gap (nspin=1, nelec=704 -> 352 occupied bands):
  per k-point: VBM_k = band[352], CBM_k = band[353];  VBM = max_k VBM_k, CBM = min_k CBM_k.
  gap = CBM - VBM. Identical settings for crystal + amorphous so PBE error cancels in the SHIFT.
  dEg_disorder = <gap>_amorph - gap_crystal (negative = narrowing).
Naive occupation gap counts localized tail states -> a LOWER bound on the mobility gap
(over-counts narrowing); IPR mobility edge is the refinement. Urbach E_U = valence DOS-tail slope.
Writes results/tier3/disorder_deg.json.
"""
import json
import re
from pathlib import Path
import numpy as np

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
NOCC = 352            # occupied bands (nelec 704 / 2, nspin=1)
SIGMA = 0.05


def per_k_eigs(prefix):
    """Return list of sorted eigenvalue arrays, one per k-point (ev)."""
    p = CC / f"dos_{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    txt = p.read_text()
    # take the LAST scf occurrence of the band listing (after convergence)
    blocks = re.split(r"\n\s+k =", txt)
    ks = []
    for b in blocks[1:]:
        # eigenvalues are the float tokens on the lines right after the k header, before a blank line
        m = re.match(r"[^\n]*\n\n?(.*?)(?:\n\s*\n|\n\s+k =|\Z)", b, re.S)
        if not m:
            continue
        vals = [float(x) for x in re.findall(r"[-]?\d+\.\d\d\d\d", m.group(1))]
        if len(vals) >= NOCC + 1:
            ks.append(np.sort(np.array(vals)))
    # keep only the last full set (nk k-points) — verbosity high prints them once at the end
    return ks[-16:] if len(ks) > 16 else ks


def gap(ks):
    if not ks:
        return None
    vbm = max(k[NOCC - 1] for k in ks)      # highest occupied over k
    cbm = min(k[NOCC] for k in ks)          # lowest unoccupied over k
    return {"VBM": round(float(vbm), 3), "CBM": round(float(cbm), 3), "gap": round(float(cbm - vbm), 3)}


def urbach(ks, VBM):
    allv = np.concatenate([k[:NOCC] for k in ks])      # occupied states
    tail = allv[(allv < VBM + 0.05) & (allv > VBM - 1.5)]
    if len(tail) < 6:
        return None
    grid = np.arange(VBM - 1.2, VBM, 0.02)
    dos = np.array([np.sum(np.exp(-0.5 * ((g - tail) / SIGMA) ** 2)) for g in grid])
    dos = np.maximum(dos, 1e-6)
    m = (grid > VBM - 0.9) & (grid < VBM - 0.1)
    if m.sum() < 4:
        return None
    slope = np.polyfit(grid[m], np.log(dos[m]), 1)[0]
    return round(1000.0 / slope, 1) if slope > 0 else None      # meV


def main():
    ck = per_k_eigs("crystal")
    cg = gap(ck) if ck else None
    if not cg:
        print("crystal DOS not ready"); return
    amorph = []
    for f in sorted(CC.glob("dos_cell_*.out")):
        name = f.stem.replace("dos_", "")
        ks = per_k_eigs(name)
        g = gap(ks) if ks else None
        if g:
            amorph.append({"cell": name, **g, "E_U_meV": urbach(ks, g["VBM"])})
    if not amorph:
        print("no amorphous DOS ready"); return
    gaps = np.array([a["gap"] for a in amorph])
    eus = np.array([a["E_U_meV"] for a in amorph if a["E_U_meV"] is not None])
    out = {
        "method": "occupation-based fundamental gap, nspin=1, PBE, identical settings",
        "crystal_gap_PBE_eV": cg["gap"], "crystal_VBM": cg["VBM"], "crystal_CBM": cg["CBM"],
        "n_amorphous_cells": len(amorph),
        "amorphous_gap_mean_eV": round(float(gaps.mean()), 3),
        "amorphous_gap_std_eV": round(float(gaps.std()), 3),
        "dEg_disorder_eV": round(float(gaps.mean() - cg["gap"]), 3),
        "dEg_disorder_sem_eV": round(float(gaps.std() / np.sqrt(len(gaps))), 3),
        "Urbach_E_U_meV": round(float(eus.mean()), 1) if len(eus) else None,
        "per_cell": amorph,
        "caveats": "naive occupation gap = LOWER bound (counts localized tail states); "
                   "IPR mobility edge + HSE spot-check are refinements. Fixed-cell quench = "
                   "crystal density (NPT is the density refinement). Thermal/ZPR (~0.19 eV) "
                   "separate. First-of-kind a-Ga2O3 structural narrowing.",
    }
    (PROJ / "results/tier3/disorder_deg.json").write_text(json.dumps(out, indent=2))
    print(f"crystal PBE gap={cg['gap']} eV | amorph={out['amorphous_gap_mean_eV']}+/-{out['amorphous_gap_std_eV']}"
          f" | dEg_disorder={out['dEg_disorder_eV']}+/-{out['dEg_disorder_sem_eV']} eV"
          f" | E_U={out['Urbach_E_U_meV']} meV (n={len(amorph)})")


if __name__ == "__main__":
    main()
