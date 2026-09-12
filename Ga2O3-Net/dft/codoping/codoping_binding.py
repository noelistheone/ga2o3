"""V62 co-doping: MACE-MP-0 dopant-PAIR binding (association) energies (the "DFT-like compound computation").

Per research Part B.1: universal MLIPs (MACE-MP-0) are reliable for NEUTRAL relaxation + total-energy
DIFFERENCES of defect supercells (not for charged formation energies -> HSE06 stays truth). The non-
additive binding term delta is "the entire scientific point" of co-doping (research C2b), so we compute it.

Physical quantity (robust, reference-free): the ASSOCIATION energy
    E_assoc(sep) = E(A_Ga + B_Ga at separation `sep`) - E(A_Ga + B_Ga at MAX separation)
Both cells have identical composition (Ga30 A B O48), so the difference is PURELY the A-B interaction.
    E_assoc(nearest) < 0  => the co-dopants ATTRACT and form a bound complex (non-dilute; delta matters)
    E_assoc(nearest) ~ 0  => independent defects (dilute approximation exact; solver as-is is correct)
We also report the standard pair binding E_bind = E(AB_near)+E(host)-E(A)-E(B) as a cross-check.

2x2x2 supercell (Ga32O48, 80 atoms). FIRE relax, fmax=0.05. Writes dft/codoping/results/binding_energies.{json,csv}.
Usage: python dft/codoping/codoping_binding.py --gpu 0
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
BASE_CIF = PROJ / "data/structures/Ga2O3_base.cif"
OUT = PROJ / "dft/codoping/results"
OUT.mkdir(parents=True, exist_ok=True)

# donor+acceptor compensation pairs (priority) + donor+donor / acceptor+acceptor contrasts
PAIRS = [("Mg", "Sn"), ("Zn", "Sn"), ("Mg", "Si"), ("Mg", "Ge"), ("Zn", "Si"),
         ("Sn", "Si"), ("Mg", "Zn")]


def build_host():
    from pymatgen.core import Structure
    base = Structure.from_file(str(BASE_CIF))
    sc = base.copy(); sc.make_supercell([2, 2, 2])
    return sc


def ga_indices(sc):
    return [i for i, sp in enumerate(sc.species) if sp.symbol == "Ga"]


def relax(struct, calc, fmax=0.05, max_steps=200):
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(struct); atoms.calc = calc
    t0 = time.time()
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return float(atoms.get_potential_energy()), bool(opt.converged()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
    from mace.calculators import mace_mp
    dev = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    print(f"Loading MACE-MP-0 (medium-mpa-0) on {dev} ...", flush=True)
    calc = mace_mp(model="medium-mpa-0", dispersion=False, default_dtype="float32", device=dev)
    print("MACE loaded.", flush=True)

    sc = build_host()
    gas = ga_indices(sc)
    i_A = gas[0]
    # rank the other Ga sites by PBC distance from i_A -> pick nearest / middle / farthest for B
    dists = sorted(((sc.get_distance(i_A, j), j) for j in gas if j != i_A))
    picks = [("near", dists[0]), ("mid", dists[len(dists) // 2]), ("far", dists[-1])]
    seps = {name: dict(j=j, d_AB=round(d, 3)) for name, (d, j) in picks}
    print(f"host Ga32O48 {len(sc)} atoms; A-site idx {i_A}; B separations: "
          f"{[(k, v['d_AB']) for k, v in seps.items()]}", flush=True)

    # relax pristine host + single dopants (cached per element) for the cross-check E_bind
    energies = {}
    print("[host] relaxing pristine Ga32O48 ...", flush=True)
    E_host, c_host, t = relax(sc, calc, args.fmax, args.max_steps)
    print(f"  E_host={E_host:.3f} eV ({'conv' if c_host else 'NOT'}, {t:.0f}s)", flush=True)

    def single(elem):
        if elem in energies:
            return energies[elem]
        s = sc.copy(); s.replace(i_A, elem)
        E, c, t = relax(s, calc, args.fmax, args.max_steps)
        energies[elem] = E
        print(f"  [single {elem}] E={E:.3f} eV ({'conv' if c else 'NOT'}, {t:.0f}s)", flush=True)
        return E

    rows = []
    for (A, B) in PAIRS:
        print(f"\n=== pair {A}+{B} ===", flush=True)
        E_A = single(A); E_B = single(B)
        e_by_sep = {}
        for name, info in seps.items():
            s = sc.copy(); s.replace(i_A, A); s.replace(info["j"], B)
            E, c, t = relax(s, calc, args.fmax, args.max_steps)
            e_by_sep[name] = E
            print(f"  [{A}+{B} {name} d={info['d_AB']}A] E={E:.3f} eV ({'conv' if c else 'NOT'}, {t:.0f}s)", flush=True)
        E_far = e_by_sep["far"]
        E_assoc_near = e_by_sep["near"] - E_far          # < 0 => bound complex
        E_assoc_mid = e_by_sep["mid"] - E_far
        E_bind = e_by_sep["near"] + E_host - E_A - E_B    # standard pair binding (near config)
        rows.append(dict(
            A=A, B=B, d_near=seps["near"]["d_AB"], d_mid=seps["mid"]["d_AB"], d_far=seps["far"]["d_AB"],
            E_near=e_by_sep["near"], E_mid=e_by_sep["mid"], E_far=E_far,
            E_assoc_near_eV=E_assoc_near, E_assoc_mid_eV=E_assoc_mid, E_bind_eV=E_bind,
            verdict=("BOUND (attract)" if E_assoc_near < -0.10 else
                     "REPEL" if E_assoc_near > 0.10 else "~independent (dilute OK)")))
        pd.DataFrame(rows).to_csv(OUT / "binding_energies.csv", index=False)

    rep = dict(supercell="Ga32O48 (2x2x2)", A_site_idx=i_A, separations=seps,
               E_host_eV=E_host, single_dopant_E_eV=energies, pairs=rows,
               note=("E_assoc = E(near)-E(far), pure A-B interaction (neutral, MACE-MP-0). Negative => "
                     "bound complex => non-additive delta matters; ~0 => dilute independent-defect "
                     "approximation is exact. Charged-defect binding needs HSE06 (research B.1)."))
    (OUT / "binding_energies.json").write_text(json.dumps(rep, indent=2))
    print("\n=== summary (association / binding energies, eV) ===")
    print(f"{'pair':8s} {'d_near':>7s} {'E_assoc_near':>13s} {'E_assoc_mid':>12s} {'E_bind':>9s}  verdict")
    for r in rows:
        print(f"{r['A']+'+'+r['B']:8s} {r['d_near']:7.2f} {r['E_assoc_near_eV']:13.3f} "
              f"{r['E_assoc_mid_eV']:12.3f} {r['E_bind_eV']:9.3f}  {r['verdict']}")
    print(f"\nsaved -> {OUT}/binding_energies.{{json,csv}}")


if __name__ == "__main__":
    main()
