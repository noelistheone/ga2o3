"""V62 co-doping: donor-acceptor PAIR formation via mass action (the non-additive delta; design phase62 sec 6).

The MACE-MP-0 binding computation (dft/codoping/codoping_binding.py) shows donor+acceptor co-dopants
ATTRACT (E_bind ~ -1 eV) and like-charge pairs REPEL. So the dilute independent-defect picture is
incomplete for donor-acceptor co-doping: a fraction of the dopants associate into BOUND NEUTRAL pairs
(A^+ ... B^-) that mutually compensate and are removed from the charged (Fermi-level-setting) budget.

Equilibrium partition (ideal-dilute law of mass action), site fractions per Ga site:
    A_free + B_free  <==>  (A-B)_pair          K = Z * exp(-E_bind / kT)     (E_bind < 0 => bound)
    x_AB = K * (x_A - x_AB) * (x_B - x_AB)
=>  K x_AB^2 - (1 + K(x_A + x_B)) x_AB + K x_A x_B = 0
    x_AB = [ (1 + K(x_A+x_B)) - sqrt( (1+K(x_A+x_B))^2 - 4 K^2 x_A x_B ) ] / (2K)   (physical root <= min)

x_AB -> min(x_A, x_B) as K -> inf (strong binding => the minority dopant pairs up completely). The FREE
concentrations x_A - x_AB, x_B - x_AB are what enter the charge-neutrality solver; the neutral pairs do not.

Z = configurational/coordination factor (number of adjacent cation sites available to pair, ~6-12 in the
beta-Ga2O3 cation sublattice; default 8, a documented approximation). Caveats (honest): MACE E_bind is the
NEUTRAL-supercell binding (the fully-ionized A^+/B^- Coulomb pair is partly screened in a neutral cell, so
this is a LOWER bound on the true binding -> HSE06 needed for the charged value, research B.1); and sputter
growth is low-T / non-equilibrium, so kinetic trapping may freeze a less-paired distribution than this
equilibrium limit. Differentiable (torch) so it composes with the solver. float64 internally.
"""
from __future__ import annotations
import torch
from src.models.defect_equilibrium import KB_EV


def pair_partition(c_A, c_B, E_bind_eV, T_K, Z: float = 8.0):
    """Equilibrium free vs bound partition for a donor-acceptor co-dopant pair.

    Args (all broadcastable torch tensors or floats):
        c_A, c_B:   total cation fractions of the two dopants [0,1].
        E_bind_eV:  pair binding energy [eV] (negative = bound; from MACE/HSE06).
        T_K:        temperature [K].
        Z:          configurational coordination factor.
    Returns (c_A_free, c_B_free, c_pair) as tensors; bound pairs are neutral (excluded from charge balance).
    """
    c_A = torch.as_tensor(c_A, dtype=torch.float64)
    c_B = torch.as_tensor(c_B, dtype=torch.float64)
    E_bind = torch.as_tensor(E_bind_eV, dtype=torch.float64)
    T = torch.as_tensor(T_K, dtype=torch.float64)
    kT = (KB_EV * T).clamp_min(1e-4)
    K = Z * torch.exp(-E_bind / kT)                       # association constant (large if bound)
    s = c_A + c_B
    b = 1.0 + K * s
    disc = (b * b - 4.0 * K * K * c_A * c_B).clamp_min(0.0)
    x_AB = (b - torch.sqrt(disc)) / (2.0 * K + 1e-300)
    x_AB = x_AB.clamp(min=torch.zeros_like(x_AB), max=torch.minimum(c_A, c_B))
    return (c_A - x_AB).clamp_min(0.0), (c_B - x_AB).clamp_min(0.0), x_AB


def bound_fraction(c_A, c_B, E_bind_eV, T_K, Z: float = 8.0):
    """Fraction of the minority dopant locked in bound neutral pairs (0..1)."""
    cAf, cBf, xAB = pair_partition(c_A, c_B, E_bind_eV, T_K, Z=Z)
    minc = torch.minimum(torch.as_tensor(c_A, dtype=torch.float64),
                         torch.as_tensor(c_B, dtype=torch.float64)).clamp_min(1e-30)
    return (xAB / minc).item() if xAB.dim() == 0 else (xAB / minc)
