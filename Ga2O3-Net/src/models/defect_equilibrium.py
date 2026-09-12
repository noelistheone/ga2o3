"""V61 core — a DIFFERENTIABLE, self-consistent defect-equilibrium solver for β-Ga2O3.

This is the novel piece of V61 (design: docs/phase61_v61_design.md §2.1). It fills the literature
gap (research Brief 3): no published model embeds a charge-neutrality / Fermi-self-consistency solver
*inside* a differentiable training loop. Given per-charge formation enthalpies (which a tiny NN head
may correct, bounded — Rule 2) plus (T, pO2, dopant charge), it:

  1. builds the charge-neutrality residual  F(E_F) = p - n + Σ_q q·[V_O^q] + z_dop·c_dop   (Eq. E1)
  2. solves  F(E_F*) = 0  for the Fermi level E_F* by robust bisection (strictly-monotone ⇒ unique root)
  3. RE-ATTACHES exact gradients via the implicit function theorem (one Newton correction at the
     detached root — DEQ / Blondel et al. 2105.15183), so autograd flows to the formation-energy head
     with NO unrolled-iteration memory.

Outputs log10[V_O], E_F*, and the charge-balance residual (a free physics-faithfulness diagnostic).
Counterfactual = re-solve at a changed knob ⇒ EXACT thermodynamic response (the law is the equations,
not a learned approximation). By construction this guarantees charge neutrality, Fermi self-consistency,
the pO2 power law (−1/6 self-consistent / −1/2 frozen regimes emerge naturally from μ_O), V_O deep-donor
behaviour, and the bulk-equilibrium sign (acceptor→E_F↓→[V_O]↑).

Physics constants match src/data/kroger_synthetic.py / src/data/dcwm_transitions.py.
All-torch, runs on GPU, float64 internally for solver stability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ── physical constants (match the project closed form) ────────────────────────
KB_EV = 8.617333262e-5          # Boltzmann [eV/K]
LN10 = math.log(10.0)
EG_GA2O3 = 4.85                 # HSE06 gap [eV]
N_O_SITES = 2.85e22             # O-anion site density [cm^-3]
N_GA_SITES = N_O_SITES * 2.0 / 3.0   # Ga cation site density [cm^-3] (Ga2O3 stoichiometry)
# effective DOS at 300 K (Ga2O3: m_e*~0.28 ⇒ N_C~3.7e18; heavy/flat holes ⇒ N_V~4e19), ∝ T^1.5
N_C0 = 3.7e18
N_V0 = 4.0e19
# native V_O formation enthalpies at E_F = VBM, pO2 = 1 atm (HSE06 Lyons 2022), per charge state
NATIVE_VO_DHF = {0: 3.5, 1: 1.8, 2: 0.3}


def _carrier_dos(T_K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (T_K / 300.0).clamp_min(1e-3) ** 1.5
    return N_C0 * s, N_V0 * s


def _residual(E_F: torch.Tensor, dH_eff: torch.Tensor, q_vals: torch.Tensor,
              kT: torch.Tensor, N_C: torch.Tensor, N_V: torch.Tensor,
              z_dop: torch.Tensor, c_dop: torch.Tensor, E_D: torch.Tensor, E_A: torch.Tensor,
              E_g: float, g_D: float = 2.0, g_A: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge-neutrality residual F(E_F) [B] and the per-charge V_O concentrations [B,Q].

    [V_O^q] uses the site-limited (Fermi-Dirac) occupancy N_O/(1+exp(E_f^q/kT)) — the proper
    dilute->saturated form: bounded by N_O, smooth, overflow-safe, ~ Boltzmann N_O*exp(-E_f/kT)
    when dilute. E_f^q(E_F) = dH_eff^q + q*E_F.

    Dopants are SELF-LIMITING via their own transition level (NOT fixed fully-ionized charge):
      donor    ionized fraction f+ = 1/(1 + g_D*exp((E_F - E_D)/kT))   (de-ionizes as E_F->CBM)
      acceptor ionized fraction f- = 1/(1 + g_A*exp((E_A - E_F)/kT))   (Mg is a DEEP acceptor)
    This keeps the root inside the gap at any concentration and conserves charge by construction
    (no spurious boundary pin / large residual at few-% doping). F stays strictly decreasing in E_F.
    """
    kTb = kT.view(-1, 1)
    ef_q = dH_eff + q_vals.view(1, -1) * E_F.view(-1, 1)          # [B,Q]
    occ = torch.sigmoid(-ef_q / kTb)                             # = 1/(1+exp(ef/kT)), in (0,1)
    conc = N_O_SITES * occ                                        # [B,Q]
    defect_charge = (q_vals.view(1, -1) * conc).sum(dim=1)        # [B]
    n = N_C * torch.exp(-(E_g - E_F) / kT)                        # arg <= 0 (E_F<=E_g) -> safe
    p = N_V * torch.exp(-E_F / kT)                                # arg <= 0 (E_F>=0) -> safe
    f_don = 1.0 / (1.0 + g_D * torch.exp((E_F - E_D) / kT))       # ionized donor fraction in (0,1)
    f_acc = 1.0 / (1.0 + g_A * torch.exp((E_A - E_F) / kT))       # ionized acceptor fraction in (0,1)
    is_don = (z_dop > 0).to(E_F.dtype)
    is_acc = (z_dop < 0).to(E_F.dtype)
    dop_charge = c_dop * (is_don * f_don - is_acc * f_acc)        # [B] self-limiting ionized charge
    F = p - n + defect_charge + dop_charge                       # [B]
    return F, conc


def solve_equilibrium(dH_f: torch.Tensor, T_K: torch.Tensor, log_pO2: torch.Tensor,
                      z_dop: torch.Tensor, c_dop_frac: torch.Tensor,
                      q_vals: torch.Tensor | None = None,
                      e_bind_donor: float = 0.05, e_bind_acceptor: float = 1.3,
                      E_g: float = EG_GA2O3, n_bisect: int = 60) -> dict:
    """Differentiable self-consistent defect equilibrium.

    Args:
        dH_f:       [B,Q] formation enthalpies at E_F=VBM, pO2=1atm [eV] (the head may correct these).
        T_K:        [B] temperature [K].
        log_pO2:    [B] log10(pO2/atm) (0 = 1 atm, negative = reducing).
        z_dop:      [B] net dopant charge sign (+1 donor, -1 acceptor, 0 isovalent/undoped).
        c_dop_frac: [B] dopant cation fraction [0,1] (-> c_dop = c_frac*N_Ga).
        q_vals:     [Q] V_O charge states (default [0,1,2]).
        e_bind_donor:    shallow-donor binding below CBM [eV] (E_D = E_g - this).
        e_bind_acceptor: acceptor level above VBM [eV] (Mg ~1.3, a deep acceptor).

    Returns dict(log10_VO [B], E_F [B], residual [B], conc [B,Q]) with exact implicit gradients
    flowing to dH_f (and to log_pO2 / T_K / c_dop_frac if they require grad).
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

    kT = (KB_EV * T_K).clamp_min(1e-4)
    N_C, N_V = _carrier_dos(T_K)
    c_dop = c_dop_frac.clamp_min(0.0) * N_GA_SITES               # dopant density [cm^-3]
    E_D = torch.full_like(T_K, float(E_g - e_bind_donor))        # donor level (below CBM)
    E_A = torch.full_like(T_K, float(e_bind_acceptor))           # acceptor level (above VBM)
    # mu_O(T,pO2) shifts the V_O formation enthalpy equally for all charges:
    #   dH_eff^q = dH_f^q + 1/2 k_B T ln(pO2)  => the -1/2*log10 pO2 Brouwer term, now inside the loop.
    mu_shift = 0.5 * kT * (LN10 * log_pO2)                       # [B]
    dH_eff = dH_f + mu_shift.view(-1, 1)                         # [B,Q]

    # ── 1. robust bisection for the unique root in the physical gap [0, E_g] ───
    # F is strictly DECREASING in E_F (p↓, n↑, donor V_O charge↓) ⇒ unique root or a boundary pin:
    #   F(0) ≤ 0 → root ≤ 0 (over-doped acceptor)        ⇒ pin E_F = 0
    #   F(E_g) ≥ 0 → root ≥ E_g (over-doped donor; degenerate, Boltzmann invalid) ⇒ pin E_F = E_g
    # else standard bisection with the guaranteed bracket F(0) > 0 > F(E_g).
    with torch.no_grad():
        lo = torch.zeros_like(T_K)
        hi = torch.full_like(T_K, float(E_g))
        f0, _ = _residual(lo, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
        fg, _ = _residual(hi, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid, _ = _residual(mid, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
            in_lower = f_mid < 0                                  # decreasing F ⇒ root in (lo, mid)
            hi = torch.where(in_lower, mid, hi)
            lo = torch.where(in_lower, lo, mid)
        E_F_d = 0.5 * (lo + hi)
        pin_lo = f0 <= 0                                          # no interior root, root ≤ 0
        pin_hi = fg >= 0                                          # no interior root, root ≥ E_g
        E_F_d = torch.where(pin_lo, torch.zeros_like(E_F_d), E_F_d)
        E_F_d = torch.where(pin_hi, torch.full_like(E_F_d, float(E_g)), E_F_d)
        pinned = pin_lo | pin_hi

    # ── 2. implicit-function-theorem gradient re-attach (one Newton step) ──────
    # For interior roots this gives dE_F*/dθ = −(∂F/∂E_F)⁻¹ ∂F/∂θ exactly. For pinned (degenerate)
    # samples we keep E_F at the boundary (no extrapolation past the gap; [V_O] is E_F-insensitive
    # there as it is neutral-V_O-dominated), so the Newton term is masked off.
    with torch.enable_grad():                                    # works even inside an outer no_grad()
        ef_g = E_F_d.detach().clone().requires_grad_(True)
        F_for_g, _ = _residual(ef_g, dH_eff.detach(), q_vals, kT, N_C, N_V,
                               z_dop.detach(), c_dop.detach(), E_D, E_A, E_g)
        g = torch.autograd.grad(F_for_g.sum(), ef_g)[0].detach()  # dF/dE_F at root [B]
    g = torch.where(g.abs() < 1e-30, torch.full_like(g, -1e-30), g)
    F_theta, _ = _residual(E_F_d.detach(), dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)  # carries ∂F/∂θ
    newton = torch.where(pinned, torch.zeros_like(F_theta), F_theta / g)
    E_F_star = (E_F_d.detach() - newton).clamp(0.0, float(E_g))  # stay in the physical gap

    # ── 3. concentrations / outputs at the self-consistent E_F* ────────────────
    _, conc = _residual(E_F_star, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
    VO_total = conc.sum(dim=1).clamp_min(1.0)
    log10_VO = torch.log10(VO_total)
    residual, _ = _residual(E_F_star.detach(), dH_eff.detach(), q_vals, kT, N_C, N_V, z_dop.detach(), c_dop.detach(), E_D, E_A, E_g)
    return dict(log10_VO=log10_VO.to(in_dtype), E_F=E_F_star.to(in_dtype),
                residual=residual.to(in_dtype), conc=conc.to(in_dtype))


@dataclass
class DefectEquilibriumSolver:
    """Thin config-holding wrapper. Holds the default V_O charge states; callable like the function."""
    q_vals: tuple = (0.0, 1.0, 2.0)
    n_bisect: int = 60
    E_g: float = EG_GA2O3

    def __call__(self, dH_f, T_K, log_pO2, z_dop, c_dop_frac):
        q = torch.tensor(self.q_vals, device=dH_f.device)
        return solve_equilibrium(dH_f, T_K, log_pO2, z_dop, c_dop_frac,
                                 q_vals=q, E_g=self.E_g, n_bisect=self.n_bisect)
