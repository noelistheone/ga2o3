"""R6 reviewer #2 item 1: anisotropic (Kumagai-Oba-type) point-charge correction vs the
direction-averaged Makov-Payne monopole used in production.

Method: exact Ewald Madelung energy of a point charge q in a neutralizing background for the
actual 80-atom supercell, evaluated (a) with isotropic epsilon (validation vs the production
MP number), and (b) with the full anisotropic static dielectric tensor via the standard
transformation  E_aniso = E_vac(eps^{-1/2} L) / sqrt(det eps).

Tensor sources (verified 2026-07-09): Gopalan et al., APL 117, 252103 (2020) THz experiment —
diagonal in the (a*, b, c) frame: (10.05, 10.6, 12.4); Fiedler et al. ECS JSSST 8, Q3083 (2019)
orientation-averaged 11.2 +/- 0.2; MP mp-886 DFPT total tensor eig (11.32, 11.51, 13.82).
Frame mapping: spglib std_rotation_matrix + comparison of the standardized conventional cell.
Writes results/tier3/efnv_aniso_check.json."""
import json
from pathlib import Path
import numpy as np
from numpy.linalg import inv, det, norm
from scipy.special import erfc
import ase.io
import spglib

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
HARTREE = 27.211386245988
BOHR = 0.529177210903


def madelung_vac(L_ang, q=1.0):
    """Vacuum Ewald Madelung ENERGY (eV) of point charge q + jellium in lattice L (Angstrom rows)."""
    L = np.asarray(L_ang) / BOHR                    # bohr
    V = abs(det(L))
    G = 2 * np.pi * inv(L).T
    eta = (np.pi / V ** (2.0 / 3.0)) ** 0.5         # standard split
    # real-space
    nmax = int(np.ceil(6.0 / (eta * min(norm(L, axis=1))))) + 2
    rs = 0.0
    rng = range(-nmax, nmax + 1)
    for i in rng:
        for j in rng:
            for k in rng:
                if i == j == k == 0:
                    continue
                R = i * L[0] + j * L[1] + k * L[2]
                r = norm(R)
                rs += erfc(eta * r) / r
    # reciprocal
    gmax = int(np.ceil(6.0 * eta * max(norm(inv(L), axis=0)) * V ** (1 / 3.0))) + 6
    ks = 0.0
    for i in range(-gmax, gmax + 1):
        for j in range(-gmax, gmax + 1):
            for k in range(-gmax, gmax + 1):
                if i == j == k == 0:
                    continue
                g = i * G[0] + j * G[1] + k * G[2]
                g2 = g @ g
                ks += 4 * np.pi / V * np.exp(-g2 / (4 * eta ** 2)) / g2
    vm = rs + ks - 2 * eta / np.sqrt(np.pi) - np.pi / (eta ** 2 * V)
    return 0.5 * q * q * vm * HARTREE               # eV (negative)


def corr_iso(L, q, eps):
    return -madelung_vac(L, q) / eps


def corr_aniso(L, q, eps_tensor):
    w, U = np.linalg.eigh(eps_tensor)
    eps_mhalf = U @ np.diag(w ** -0.5) @ U.T
    Lt = np.asarray(L) @ eps_mhalf.T                # rows transformed: R' = eps^{-1/2} R
    return -madelung_vac(Lt, q) / np.sqrt(det(eps_tensor))


atoms = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")
L80 = np.array(atoms.get_cell())
cell = (L80, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
ds = spglib.get_symmetry_dataset(cell, symprec=1e-3)
Rstd = np.array(ds.std_rotation_matrix if hasattr(ds, "std_rotation_matrix")
                else ds["std_rotation_matrix"])
sg = ds.international if hasattr(ds, "international") else ds["international"]
# spglib standardized monoclinic (C2/m, unique axis b): a along x-ish, b along y, c in x-z plane.
# Gopalan tensor is diagonal in (a*, b, c). In the standardized Cartesian frame: b || y.
# Build the tensor in the standardized frame from the standardized conventional lattice:
std_lat = np.array(ds.std_lattice if hasattr(ds, "std_lattice") else ds["std_lattice"])
a_v, b_v, c_v = std_lat
b_hat = b_v / norm(b_v)
c_hat = c_v / norm(c_v)
astar = np.cross(b_v, c_v); astar /= norm(astar)    # a* is perpendicular to b and c
E_GOP = np.diag([10.05, 10.6, 12.4])                # in (a*, b, c) orthonormal-ish frame
# orthonormal frame: e1=a*, e3=c, e2=b (b is orthogonal to both in monoclinic)
F = np.vstack([astar, b_hat, c_hat]).T              # columns = frame vectors in std Cartesian
# c and a* are orthogonal? a* ⟂ c by construction (a* = b x c / |..|). b ⟂ a*, b ⟂ c (monoclinic).
eps_std = F @ E_GOP @ F.T
# rotate into OUR Cartesian frame: std frame = Rstd @ our frame  =>  eps_our = Rstd^T eps_std Rstd
eps_our = Rstd.T @ eps_std @ Rstd
# MP-DFPT theory tensor (eigenvalues; same principal frame assumption)
E_MPT = np.diag([11.32, 11.51, 13.82])
eps_our_dfpt = Rstd.T @ (F @ E_MPT @ F.T) @ Rstd

q = 2.0
res = {
    "spacegroup": str(sg),
    "orthogonality_check": {"bdotc": float(b_v @ c_v / (norm(b_v) * norm(c_v)))},
    "E_iso_eps10.2_eV": round(corr_iso(L80, q, 10.2), 4),
    "production_MP_eV": 0.847,
    "E_iso_eps11.2_Fiedler_eV": round(corr_iso(L80, q, 11.2), 4),
    "E_aniso_Gopalan_eV": round(corr_aniso(L80, q, eps_our), 4),
    "E_aniso_MP_DFPT_eV": round(corr_aniso(L80, q, eps_our_dfpt), 4),
}
res["delta_aniso_vs_production_eV"] = round(res["E_aniso_Gopalan_eV"] - res["production_MP_eV"], 4)
res["delta_level_eV"] = round(res["delta_aniso_vs_production_eV"] / 2.0, 4)
res["note"] = ("exact Ewald point-charge+jellium; anisotropic via E_vac(eps^-1/2 L)/sqrt(det eps). "
               "Gopalan tensor (a*,b,c)=(10.05,10.6,12.4) exp; Fiedler avg 11.2; MP mp-886 DFPT "
               "eig (11.32,11.51,13.82). Level shift = delta/2 for the (2+/0) level.")
(PROJ / "results/tier3/efnv_aniso_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
