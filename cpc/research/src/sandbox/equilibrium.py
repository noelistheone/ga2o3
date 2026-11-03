"""Self-consistent defect-equilibrium solver for β-Ga2O3 (Ga2O3-Sandbox core).

PROVENANCE: ported verbatim-physics from Ga2O3-Net src/models/defect_equilibrium.py (V61,
2026-06) — a differentiable charge-neutrality solver (bisection root + implicit-function-theorem
gradient re-attach). Physics unchanged; additions here are OUTPUT helpers only:

  * free-carrier densities n, p at the self-consistent Fermi level (Boltzmann statistics — the
    V61 regime of validity; the degenerate Fermi-Dirac variant is ported separately when needed),
  * ionized-dopant density and per-charge V_O breakdown in the returned dict,
  * energetics are read from data/energetics/*.json (provenance-tracked), never hard-coded —
    the module-level NATIVE_VO_DHF constant is kept ONLY as the documented default and must
    match data/energetics/native_VO_hse06_anchor.json (checked by scripts/phase0_brouwer_demo.py).

Charge-neutrality residual:  F(E_F) = p − n + Σ_q q·[V_O^q] + z_dop·c_dop·f_ion(E_F)
[V_O^q] site-limited: N_O/(1+exp(E_f^q/kT)), E_f^q(E_F) = ΔH_f^q + q·E_F + ½kT·ln(pO2).
Dopants self-limiting via their own level (donor de-ionizes toward CBM, acceptor toward VBM).
F is strictly decreasing in E_F ⇒ unique interior root or a boundary pin.

All-torch, float64 internally, runs on GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ── physical constants (identical to Ga2O3-Net V61 / kroger_synthetic.py) ─────
KB_EV = 8.617333262e-5          # Boltzmann [eV/K]
LN10 = math.log(10.0)
EG_GA2O3 = 4.85                 # HSE06 gap [eV]
N_O_SITES = 2.85e22             # O-anion site density [cm^-3]
N_GA_SITES = N_O_SITES * 2.0 / 3.0   # Ga cation site density [cm^-3]
N_C0 = 3.7e18                   # effective conduction DOS at 300 K (m_e* ~ 0.28), ∝ T^1.5
N_V0 = 4.0e19                   # effective valence DOS at 300 K (heavy/flat holes), ∝ T^1.5
# Default native V_O anchor (E_F=VBM, pO2=1 atm) — MUST match data/energetics JSON (see docstring)
NATIVE_VO_DHF = {0: 3.5, 1: 1.8, 2: 0.3}


def _carrier_dos(T_K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (T_K / 300.0).clamp_min(1e-3) ** 1.5
    return N_C0 * s, N_V0 * s


def _residual(E_F: torch.Tensor, dH_eff: torch.Tensor, q_vals: torch.Tensor,
              kT: torch.Tensor, N_C: torch.Tensor, N_V: torch.Tensor,
              z_dop: torch.Tensor, c_dop: torch.Tensor, E_D: torch.Tensor, E_A: torch.Tensor,
              E_g: float, g_D: float = 2.0, g_A: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge-neutrality residual F(E_F) [B] and per-charge V_O concentrations [B,Q]."""
    kTb = kT.view(-1, 1)
    ef_q = dH_eff + q_vals.view(1, -1) * E_F.view(-1, 1)          # [B,Q]
    occ = torch.sigmoid(-ef_q / kTb)                              # 1/(1+exp(ef/kT))
    conc = N_O_SITES * occ                                        # [B,Q]
    defect_charge = (q_vals.view(1, -1) * conc).sum(dim=1)        # [B]
    n = N_C * torch.exp(-(E_g - E_F) / kT)
    p = N_V * torch.exp(-E_F / kT)
    f_don = 1.0 / (1.0 + g_D * torch.exp((E_F - E_D) / kT))       # ionized donor fraction
    f_acc = 1.0 / (1.0 + g_A * torch.exp((E_A - E_F) / kT))       # ionized acceptor fraction
    is_don = (z_dop > 0).to(E_F.dtype)
    is_acc = (z_dop < 0).to(E_F.dtype)
    dop_charge = c_dop * (is_don * f_don - is_acc * f_acc)
    F = p - n + defect_charge + dop_charge
    return F, conc


def solve_equilibrium(dH_f: torch.Tensor, T_K: torch.Tensor, log_pO2: torch.Tensor,
                      z_dop: torch.Tensor, c_dop_frac: torch.Tensor,
                      q_vals: torch.Tensor | None = None,
                      e_bind_donor: float = 0.05, e_bind_acceptor: float = 1.3,
                      E_g: float = EG_GA2O3, n_bisect: int = 60) -> dict:
    """Differentiable self-consistent defect equilibrium (V61 semantics + carrier outputs).

    Args:
        dH_f:       [B,Q] V_O formation enthalpies at E_F=VBM, pO2=1atm [eV].
        T_K:        [B] temperature [K].
        log_pO2:    [B] log10(pO2/atm).
        z_dop:      [B] dopant charge sign (+1 donor, −1 acceptor, 0 isovalent/undoped).
        c_dop_frac: [B] dopant cation fraction [0,1].
        e_bind_donor / e_bind_acceptor: dopant level position (E_D = E_g − b_D; E_A = b_A).

    Returns dict(log10_VO, E_F, residual, conc[B,Q], n, p, log10_n, log10_p, c_dop_ionized,
                 pinned) — implicit gradients flow to dH_f (and T/pO2/c if they require grad).
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
    c_dop = c_dop_frac.clamp_min(0.0) * N_GA_SITES
    E_D = torch.full_like(T_K, float(E_g - e_bind_donor))
    E_A = torch.full_like(T_K, float(e_bind_acceptor))
    # μ_O(T,pO2) shift: dH_eff^q = dH_f^q + ½ k_B T ln(pO2)  (the −½·log10 pO2 Brouwer term)
    mu_shift = 0.5 * kT * (LN10 * log_pO2)
    dH_eff = dH_f + mu_shift.view(-1, 1)

    # ── 1. robust bisection for the unique root in [0, E_g] ────────────────────
    with torch.no_grad():
        lo = torch.zeros_like(T_K)
        hi = torch.full_like(T_K, float(E_g))
        f0, _ = _residual(lo, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
        fg, _ = _residual(hi, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid, _ = _residual(mid, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
            in_lower = f_mid < 0
            hi = torch.where(in_lower, mid, hi)
            lo = torch.where(in_lower, lo, mid)
        E_F_d = 0.5 * (lo + hi)
        pin_lo = f0 <= 0
        pin_hi = fg >= 0
        E_F_d = torch.where(pin_lo, torch.zeros_like(E_F_d), E_F_d)
        E_F_d = torch.where(pin_hi, torch.full_like(E_F_d, float(E_g)), E_F_d)
        pinned = pin_lo | pin_hi

    # ── 2. implicit-function-theorem gradient re-attach (one Newton step) ──────
    with torch.enable_grad():
        ef_g = E_F_d.detach().clone().requires_grad_(True)
        F_for_g, _ = _residual(ef_g, dH_eff.detach(), q_vals, kT, N_C, N_V,
                               z_dop.detach(), c_dop.detach(), E_D, E_A, E_g)
        g = torch.autograd.grad(F_for_g.sum(), ef_g)[0].detach()
    g = torch.where(g.abs() < 1e-30, torch.full_like(g, -1e-30), g)
    F_theta, _ = _residual(E_F_d.detach(), dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
    newton = torch.where(pinned, torch.zeros_like(F_theta), F_theta / g)
    E_F_star = (E_F_d.detach() - newton).clamp(0.0, float(E_g))

    # ── 3. outputs at the self-consistent E_F* ─────────────────────────────────
    _, conc = _residual(E_F_star, dH_eff, q_vals, kT, N_C, N_V, z_dop, c_dop, E_D, E_A, E_g)
    VO_total = conc.sum(dim=1).clamp_min(1.0)
    log10_VO = torch.log10(VO_total)
    residual, _ = _residual(E_F_star.detach(), dH_eff.detach(), q_vals, kT, N_C, N_V,
                            z_dop.detach(), c_dop.detach(), E_D, E_A, E_g)
    n = (N_C * torch.exp(-(E_g - E_F_star) / kT)).clamp_min(1.0)
    p = (N_V * torch.exp(-E_F_star / kT)).clamp_min(1.0)
    f_don = 1.0 / (1.0 + 2.0 * torch.exp((E_F_star - E_D) / kT))
    f_acc = 1.0 / (1.0 + 4.0 * torch.exp((E_A - E_F_star) / kT))
    is_don = (z_dop > 0).to(E_F_star.dtype)
    is_acc = (z_dop < 0).to(E_F_star.dtype)
    c_ion = c_dop * (is_don * f_don + is_acc * f_acc)
    return dict(log10_VO=log10_VO.to(in_dtype), E_F=E_F_star.to(in_dtype),
                residual=residual.to(in_dtype), conc=conc.to(in_dtype),
                n=n.to(in_dtype), p=p.to(in_dtype),
                log10_n=torch.log10(n).to(in_dtype), log10_p=torch.log10(p).to(in_dtype),
                c_dop_ionized=c_ion.to(in_dtype), pinned=pinned)


@dataclass
class DefectEquilibriumSolver:
    """Thin config-holding wrapper; callable like the function."""
    q_vals: tuple = (0.0, 1.0, 2.0)
    n_bisect: int = 60
    E_g: float = EG_GA2O3

    def __call__(self, dH_f, T_K, log_pO2, z_dop, c_dop_frac, **kw):
        q = torch.tensor(self.q_vals, device=dH_f.device)
        return solve_equilibrium(dH_f, T_K, log_pO2, z_dop, c_dop_frac,
                                 q_vals=q, E_g=self.E_g, n_bisect=self.n_bisect, **kw)
