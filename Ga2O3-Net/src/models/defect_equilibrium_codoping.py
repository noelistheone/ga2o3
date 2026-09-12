"""V62 core — MULTI-DOPANT differentiable self-consistent defect-equilibrium solver for beta-Ga2O3.

Generalizes the single-dopant V61 solver (src/models/defect_equilibrium.py) to an arbitrary number
D of simultaneous dopant species (co-doping). Design + math proof: docs/phase62_codoping_design.md.

Central equation (Eq. C1), shared Fermi level E_F (from VBM):

    F(E_F) = p - n + Sum_q q*[V_O^q] + Sum_{i=1..D} z_i*[D_i^ion]   = 0

Each dopant i contributes z_i * c_i * f_i(E_F) where f_i is its self-limiting ionized fraction
(donor de-ionizes toward CBM, acceptor toward VBM) at the species' own transition level E_i.

KEY THEOREM (docs/phase62 sec 2.2): F is strictly DECREASING in E_F for ANY number of dopants of ANY
sign (additivity preserves monotonicity), so the unique-root bisection + implicit-function-theorem
gradient of the single-dopant solver extend verbatim. The ONLY dopant-dopant coupling is the shared
E_F (dilute non-interacting / self-consistent-Fermi approximation) -- exactly charge compensation.

Single-dopant case (D=1, default class levels) reproduces V61 bit-for-bit (regression-tested in
scripts/78_codoping_solver_smoke.py). All-torch, float64 internally, runs on GPU. Rule 1/2 preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Re-use the V61 physical constants so the two solvers stay numerically identical.
from src.models.defect_equilibrium import (  # noqa: F401
    KB_EV, LN10, EG_GA2O3, N_O_SITES, N_GA_SITES, N_C0, N_V0, NATIVE_VO_DHF, _carrier_dos,
)

# ---- per-element transition levels (eV) ---------------------------------------------------------
# Donors: level E_D below CBM (stored as binding below CBM -> E_D = E_g - bind). Acceptors: level E_A
# above VBM. Values from project HSE06 (dft/qe_hse06) + Ga2O3 defect-chemistry literature; elements
# without a dedicated calc use the CLASS DEFAULT (shallow donor 0.05 below CBM; deep acceptor 1.3 above
# VBM) -- those defaults exactly match the V61 single-dopant solver so D=1 reduces to V61. Approximate
# entries are flagged low-confidence in the design doc (sec 6).
_DONOR_BIND_BELOW_CBM = {        # E_D = E_g - this
    "Sn": 0.03, "Si": 0.03, "Ge": 0.04,        # well-established shallow donors in beta-Ga2O3
    "Ti": 0.05, "Zr": 0.05, "Hf": 0.05, "Ta": 0.05, "W": 0.05, "V": 0.05, "Nb": 0.05,
    "H": 0.04, "Sb": 0.20, "Bi": 0.30,         # Sb/Bi deeper, low-confidence
}
_ACCEPTOR_LEVEL_ABOVE_VBM = {    # E_A above VBM
    "Mg": 1.3, "Zn": 1.3, "Cu": 1.4, "Ni": 1.4, "N": 1.4, "F": 1.3,   # deep acceptors
}
_DEFAULT_DONOR_BIND = 0.05       # matches V61 e_bind_donor
_DEFAULT_ACCEPTOR_LEVEL = 1.3    # matches V61 e_bind_acceptor


def elem_transition_level(elem: str, sign: float, E_g: float = EG_GA2O3) -> float:
    """Transition level E_i (eV above VBM) for a dopant, interpreted by its carrier sign.

    sign > 0 (donor)   -> E_i = E_g - binding-below-CBM   (near CBM for a shallow donor)
    sign < 0 (acceptor)-> E_i = level-above-VBM           (1.3 eV for a deep acceptor like Mg)
    sign == 0 (isovalent) -> mid-gap placeholder (contributes zero charge; level is irrelevant).
    """
    if sign > 0:
        return float(E_g - _DONOR_BIND_BELOW_CBM.get(elem, _DEFAULT_DONOR_BIND))
    if sign < 0:
        return float(_ACCEPTOR_LEVEL_ABOVE_VBM.get(elem, _DEFAULT_ACCEPTOR_LEVEL))
    return float(0.5 * E_g)


def _residual_codoping(E_F, dH_eff, q_vals, kT, N_C, N_V,
                       z_dop, c_dop, e_level, dop_mask,
                       E_g, g_D: float = 2.0, g_A: float = 4.0):
    """Charge-neutrality residual F(E_F) [B], V_O per-charge conc [B,Q], dopant ionized frac [B,D].

    z_dop/c_dop/e_level/dop_mask are [B,D] (D = max dopant slots). c_dop is a DENSITY [cm^-3].
    Each species' ionized fraction (Fermi-Dirac, site/self-limiting):
        donor    f+ = 1/(1 + g_D*exp((E_F - E_i)/kT))     (de-ionizes as E_F -> CBM)
        acceptor f- = 1/(1 + g_A*exp((E_i - E_F)/kT))     (de-ionizes as E_F -> VBM)
    Net per-species charge = c_i * (is_don*f+ - is_acc*f-); summed over masked slots.
    F stays strictly decreasing in E_F for any D and any mix of signs (design doc sec 2.2).
    """
    kTb = kT.view(-1, 1)                                         # [B,1]
    # --- V_O (host oxygen vacancy), identical to V61 ---
    ef_q = dH_eff + q_vals.view(1, -1) * E_F.view(-1, 1)         # [B,Q]
    occ = torch.sigmoid(-ef_q / kTb)
    conc = N_O_SITES * occ                                       # [B,Q]
    defect_charge = (q_vals.view(1, -1) * conc).sum(dim=1)       # [B]
    # --- carriers ---
    n = N_C * torch.exp(-(E_g - E_F) / kT)
    p = N_V * torch.exp(-E_F / kT)
    # --- dopants: masked sum over D species, each with its own sign + transition level ---
    EF_b = E_F.view(-1, 1)                                       # [B,1]
    f_don = 1.0 / (1.0 + g_D * torch.exp((EF_b - e_level) / kTb))   # [B,D] ionized donor fraction
    f_acc = 1.0 / (1.0 + g_A * torch.exp((e_level - EF_b) / kTb))   # [B,D] ionized acceptor fraction
    is_don = (z_dop > 0).to(E_F.dtype)
    is_acc = (z_dop < 0).to(E_F.dtype)
    dop_ion = is_don * f_don - is_acc * f_acc                    # [B,D] net ionized fraction (signed)
    dop_charge = (dop_mask * c_dop * dop_ion).sum(dim=1)         # [B]
    F = p - n + defect_charge + dop_charge                      # [B]
    return F, conc, dop_ion


def solve_equilibrium_codoping(dH_f, T_K, log_pO2, z_dop, c_dop_frac, e_level,
                               dop_mask=None, q_vals=None, E_g: float = EG_GA2O3,
                               n_bisect: int = 60) -> dict:
    """Differentiable self-consistent MULTI-DOPANT defect equilibrium.

    Args:
        dH_f:       [B,Q] V_O formation enthalpies at E_F=VBM, pO2=1atm [eV] (head may correct, Rule 2).
        T_K:        [B] temperature [K].
        log_pO2:    [B] log10(pO2/atm).
        z_dop:      [B,D] per-species net charge sign (+1 donor, -1 acceptor, 0 isovalent/pad).
        c_dop_frac: [B,D] per-species cation fraction [0,1] (-> density c_i = frac*N_Ga).
        e_level:    [B,D] per-species transition level [eV above VBM] (see elem_transition_level).
        dop_mask:   [B,D] 1.0 for real species, 0.0 for padding (default: all ones).
        q_vals:     [Q] V_O charge states (default [0,1,2]).

    Returns dict(log10_VO [B], E_F [B], residual [B], conc [B,Q], dop_ion [B,D]) with exact implicit
    gradients flowing to dH_f (and to log_pO2/T_K/c_dop_frac if they require grad).
    """
    in_dtype = dH_f.dtype
    dev = dH_f.device
    if q_vals is None:
        q_vals = torch.tensor([0.0, 1.0, 2.0], device=dev)
    q_vals = q_vals.to(dev).double()
    dH_f = dH_f.double()
    T_K = T_K.double().to(dev)
    log_pO2 = log_pO2.double().to(dev)
    z_dop = z_dop.double().to(dev)
    c_dop_frac = c_dop_frac.double().to(dev)
    e_level = e_level.double().to(dev)
    if z_dop.dim() == 1:                                         # tolerate [B] -> [B,1]
        z_dop = z_dop.view(-1, 1); c_dop_frac = c_dop_frac.view(-1, 1); e_level = e_level.view(-1, 1)
    if dop_mask is None:
        dop_mask = torch.ones_like(z_dop)
    dop_mask = dop_mask.double().to(dev)

    kT = (KB_EV * T_K).clamp_min(1e-4)
    N_C, N_V = _carrier_dos(T_K)
    c_dop = c_dop_frac.clamp_min(0.0) * N_GA_SITES              # [B,D] dopant densities [cm^-3]
    mu_shift = 0.5 * kT * (LN10 * log_pO2)                     # -1/2 log10(pO2) Brouwer term in-loop
    dH_eff = dH_f + mu_shift.view(-1, 1)                        # [B,Q]

    def resid(EF):
        return _residual_codoping(EF, dH_eff, q_vals, kT, N_C, N_V,
                                  z_dop, c_dop, e_level, dop_mask, E_g)

    # --- 1. robust bisection for the unique root in [0, E_g] (F strictly decreasing) ---
    with torch.no_grad():
        lo = torch.zeros_like(T_K)
        hi = torch.full_like(T_K, float(E_g))
        f0, _, _ = resid(lo)
        fg, _, _ = resid(hi)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid, _, _ = resid(mid)
            in_lower = f_mid < 0                                # decreasing F -> root in (lo, mid)
            hi = torch.where(in_lower, mid, hi)
            lo = torch.where(in_lower, lo, mid)
        E_F_d = 0.5 * (lo + hi)
        pin_lo = f0 <= 0
        pin_hi = fg >= 0
        E_F_d = torch.where(pin_lo, torch.zeros_like(E_F_d), E_F_d)
        E_F_d = torch.where(pin_hi, torch.full_like(E_F_d, float(E_g)), E_F_d)
        pinned = pin_lo | pin_hi

    # --- 2. implicit-function-theorem gradient re-attach (one Newton step) ---
    with torch.enable_grad():
        ef_g = E_F_d.detach().clone().requires_grad_(True)
        F_for_g, _, _ = _residual_codoping(
            ef_g, dH_eff.detach(), q_vals, kT, N_C, N_V,
            z_dop.detach(), c_dop.detach(), e_level.detach(), dop_mask.detach(), E_g)
        g = torch.autograd.grad(F_for_g.sum(), ef_g)[0].detach()
    g = torch.where(g.abs() < 1e-30, torch.full_like(g, -1e-30), g)
    F_theta, _, _ = resid(E_F_d.detach())                       # carries dF/dtheta
    newton = torch.where(pinned, torch.zeros_like(F_theta), F_theta / g)
    E_F_star = (E_F_d.detach() - newton).clamp(0.0, float(E_g))

    # --- 3. concentrations / outputs at the self-consistent E_F* ---
    _, conc, dop_ion = resid(E_F_star)
    VO_total = conc.sum(dim=1).clamp_min(1.0)
    log10_VO = torch.log10(VO_total)
    residual, _, _ = _residual_codoping(
        E_F_star.detach(), dH_eff.detach(), q_vals, kT, N_C, N_V,
        z_dop.detach(), c_dop.detach(), e_level.detach(), dop_mask.detach(), E_g)
    return dict(log10_VO=log10_VO.to(in_dtype), E_F=E_F_star.to(in_dtype),
                residual=residual.to(in_dtype), conc=conc.to(in_dtype),
                dop_ion=dop_ion.to(in_dtype))


@dataclass
class CoDopingEquilibriumSolver:
    """Thin config-holding wrapper; callable like the function."""
    q_vals: tuple = (0.0, 1.0, 2.0)
    n_bisect: int = 60
    E_g: float = EG_GA2O3

    def __call__(self, dH_f, T_K, log_pO2, z_dop, c_dop_frac, e_level, dop_mask=None):
        q = torch.tensor(self.q_vals, device=dH_f.device)
        return solve_equilibrium_codoping(dH_f, T_K, log_pO2, z_dop, c_dop_frac, e_level,
                                          dop_mask=dop_mask, q_vals=q, E_g=self.E_g,
                                          n_bisect=self.n_bisect)
