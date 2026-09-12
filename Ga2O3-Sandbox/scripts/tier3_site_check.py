"""Address the research red flag: verify Sb/Bi prefer the Ga CATION site (not the O anion
site). The (2+/0) donor assignment assumes a cation-site substitutional; if the dopant
preferred the O site it would be a different defect entirely. Fully-MACE site comparison
(consistent scale): E_f(M_Ga) vs E_f(M_O) using MACE elemental references.

  E_f(M_Ga) - E_f(M_O) = [E(M_Ga) - E(M_O)] + mu_Ga - mu_O   (mu_M cancels)
mu from MACE: mu_O = 1/2 E(O2), mu_Ga = [E(Ga2O3 fu) - 3 mu_O]/2 (O-rich);
Ga-rich uses mu_Ga = E(alpha-Ga)/atom. Negative => Ga site preferred.
"""
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import ase.io
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

PROJ = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
CC = PROJ / "dft/qe_cc"
FMAX, STEPS = 0.03, 400


def calc():
    from mace.calculators import mace_mp
    return mace_mp(model="/home/lawrence/.cache/mace/macempa0mediummodel", device="cuda",
                   default_dtype="float64")


def relax(atoms, c, cell=False):
    a = atoms.copy(); a.calc = c
    FIRE(FrechetCellFilter(a) if cell else a, logfile=None).run(fmax=FMAX, steps=STEPS)
    return float(a.get_potential_energy())


def main():
    c = calc()
    perfect = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")
    E_perf = relax(perfect, c, cell=False)
    E_fu = E_perf / 16.0
    # MACE elemental refs
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=[15, 15, 15], pbc=True)
    mu_O = 0.5 * relax(o2, c)
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    ga = AseAtomsAdaptor.get_atoms(Structure.from_spacegroup(
        "Cmce", Lattice.orthorhombic(4.5192, 7.6586, 4.5258), ["Ga"], [[0, 0.15263, 0.08128]]))
    mu_Ga_metal = relax(ga, c, cell=True) / len(ga)
    mu_Ga_Orich = (E_fu - 3 * mu_O) / 2.0

    sym = np.array(perfect.get_chemical_symbols())
    ga_idx = np.where(sym == "Ga")[0]
    o_idx = np.where(sym == "O")[0]

    out = {"mu_O_MACE": round(mu_O, 3), "mu_Ga_Orich": round(mu_Ga_Orich, 3),
           "mu_Ga_metal": round(mu_Ga_metal, 3), "dopants": {}}
    for elem in ["Sb", "Bi"]:
        # M on a Ga site (octahedral, the preferred cation site)
        a_ga = perfect.copy(); a_ga[int(ga_idx[0])].symbol = elem
        E_MGa = relax(a_ga, c)
        # M on an O site
        a_o = perfect.copy(); a_o[int(o_idx[0])].symbol = elem
        E_MO = relax(a_o, c)
        # site-preference (Ga - O); mu_M cancels. Two mu_Ga limits.
        d_Orich = (E_MGa - E_MO) + mu_Ga_Orich - mu_O
        d_Garich = (E_MGa - E_MO) + mu_Ga_metal - mu_O
        out["dopants"][elem] = {
            "E_M_Ga": round(E_MGa, 3), "E_M_O": round(E_MO, 3),
            "Ef(Ga)-Ef(O)_Orich_eV": round(d_Orich, 3),
            "Ef(Ga)-Ef(O)_Garich_eV": round(d_Garich, 3),
            "prefers": "Ga(cation) site" if max(d_Orich, d_Garich) < 0 else
                       ("O(anion) site" if min(d_Orich, d_Garich) > 0 else "condition-dependent"),
        }
        print(f"[{elem}] Ef(Ga)-Ef(O): O-rich {d_Orich:+.2f}, Ga-rich {d_Garich:+.2f} eV "
              f"-> {out['dopants'][elem]['prefers']}")
    (PROJ / "results/tier3/site_check.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/tier3/site_check.json")


if __name__ == "__main__":
    main()
