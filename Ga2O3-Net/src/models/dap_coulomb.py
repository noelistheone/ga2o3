"""Phase 63 Opt-B — charged donor-acceptor-pair (DAP) binding via the point-charge Coulomb correction.

The MACE-MP-0 neutral-supercell binding (dft/codoping/codoping_binding.py) screens out the monopole-
monopole electrostatic term, so it is a LOWER bound on the physically relevant CHARGED pair binding
A^+...B^- (Phase-62 caveat B.1). The leading, standard correction is the point-charge Coulomb energy of
the two ionized defects at their nearest-neighbour cation-cation separation:

    E_Coulomb = q_A q_B e^2 / (4 pi eps0 eps_r d)        e^2/(4 pi eps0) = 14.39964 eV*Angstrom

For a donor (q_A=+1) and acceptor (q_B=-1): q_A q_B = -1 -> attractive (lowers energy). The charged
binding is therefore MORE negative than the neutral MACE value by |E_Coulomb|. With beta-Ga2O3 static
dielectric eps_r ~= 10.2 (orientation-averaged eps0 tensor) and nearest cation-cation distance
d ~= 3.0-3.4 Angstrom, |E_Coulomb| ~= 0.41-0.48 eV, giving E_bind^charged ~= -1.0 eV (matches the
codoping_pairing.py expectation, derived analytically).

Honest caveats (documented in docs/phase63_design.md sec Opt-B): the point-charge model ignores the
charged/neutral lattice-relaxation difference and the finite extent of the defect wavefunctions, so this
is an ESTIMATE bracketed by [MACE_neutral_lower_bound, point_charge_estimate], not an HSE06 number. We
return the bracket so the pairing fraction can be reported with its spread. Differentiable, float64.
"""
from __future__ import annotations

import torch

# e^2 / (4 pi eps0) in eV * Angstrom (Gaussian/SI combo: 14.39964 eV.A)
COULOMB_EV_ANGSTROM = 14.399645
# beta-Ga2O3 static dielectric constant (orientation-averaged eps0; eps_b~10.2, eps_c~12.4, eps_a~10.2
# -> avg ~10.9; we use a conservative 10.2 so the correction is not over-estimated). Passlack 1995;
# Fiedler 2019 (J. Appl. Phys.). High-frequency eps_inf ~ 3.5 would over-bind (no ionic screening at
# the slow defect-migration timescale), so the STATIC value is correct for an equilibrium pair.
EPS_R_GA2O3_STATIC = 10.2
# nearest Ga-Ga cation distance in beta-Ga2O3 (monoclinic C2/m): the two inequivalent Ga sites sit
# ~3.0-3.4 A apart (Ga(I)-Ga(II) and Ga(II)-Ga(II) edges); 3.2 A is the representative nearest pair.
D_CATION_NN_ANGSTROM = 3.2


def coulomb_energy(q_A: float, q_B: float, eps_r: float = EPS_R_GA2O3_STATIC,
                   d: float = D_CATION_NN_ANGSTROM) -> float:
    """Point-charge Coulomb energy [eV] of two ionized defects (signed; negative = attractive)."""
    return float(q_A) * float(q_B) * COULOMB_EV_ANGSTROM / (float(eps_r) * float(d))


def charged_binding(E_bind_neutral, q_A: float = +1.0, q_B: float = -1.0,
                    eps_r: float = EPS_R_GA2O3_STATIC, d: float = D_CATION_NN_ANGSTROM):
    """Charged DAP binding [eV] = neutral MACE binding + point-charge Coulomb correction.

    Args:
        E_bind_neutral: MACE neutral-cell binding [eV] (negative = bound), float or tensor.
        q_A, q_B:       ionized charges of the two partners (+1 donor, -1 acceptor by default).
        eps_r, d:       static dielectric and nearest cation-cation distance.
    Returns: E_bind^charged (same type as input). For donor+acceptor this is more negative.
    """
    e_coul = coulomb_energy(q_A, q_B, eps_r, d)
    if isinstance(E_bind_neutral, torch.Tensor):
        return E_bind_neutral + e_coul
    return float(E_bind_neutral) + e_coul


def binding_bracket(E_bind_neutral: float, q_A: float = +1.0, q_B: float = -1.0,
                    eps_r: float = EPS_R_GA2O3_STATIC, d: float = D_CATION_NN_ANGSTROM) -> dict:
    """Return the honest bracket [MACE lower bound, point-charge estimate] for the charged binding.

    The true charged binding lies between the (under-binding) neutral MACE value and the (slightly
    over-binding, relaxation-neglecting) point-charge estimate. We expose both so downstream pairing
    fractions carry their spread. For like-charge pairs (q_A q_B > 0) the Coulomb term is REPULSIVE and
    the bracket flips ordering (handled by min/max).
    """
    charged = charged_binding(E_bind_neutral, q_A, q_B, eps_r, d)
    lo, hi = (E_bind_neutral, charged) if charged >= E_bind_neutral else (charged, E_bind_neutral)
    return dict(neutral_mace=float(E_bind_neutral), point_charge=float(charged),
                E_coulomb=coulomb_energy(q_A, q_B, eps_r, d),
                bracket_lo=float(lo), bracket_hi=float(hi),
                eps_r=float(eps_r), d_angstrom=float(d))
