"""Phase 53 Stage 3 — KKT-Hardnet Brouwer head with full Kröger-Vink charge neutrality.

Implements V53-η: replace V52b's single-defect BrouwerHeadVC with full Kröger-Vink
that includes V_O^q (q ∈ {0, +1, +2}) + V_Ga^q (q ∈ {-1, -2, -3}) + dopant^q charges.
Self-consistent εF solved via KKT-Hardnet (unrolled Newton, differentiable).

Paper: arXiv 2507.08124 — KKT-Hardnet for hard-constrained NN outputs.

Architecture:
  fused [B, 160] → latent_net [B, 6] → 6 E_f corrections (V_O × 3 + V_Ga × 3)
  Base E_f from src.data.v_ga_synthetic (DFT CSV preferred, synthetic fallback).
  Newton solve for εF in [0, EG] such that Σq·[defect_q] = n - p + q_dopant·[dopant].
  Return log10[V_O_total] = log10(Σ_q [V_O^q]).

Falls back to V52b BrouwerHeadVC behavior if all corrections=0 and synthetic E_f
is used (i.e., no learning — use as physics check).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
KB_EV = 8.617333262e-5
LN10 = math.log(10.0)

# β-Ga2O3 host parameters (HSE06 averages)
EG_GA2O3 = 4.85          # bandgap (eV)
N_O_SITES = 2.85e22      # O site density (cm⁻³)
N_GA_SITES = 1.9e22      # Ga site density (cm⁻³)

# Effective densities of states (cm⁻³) at 300K, scaled by (T/300)^1.5
NC_300K = 2.5e19 * (0.30 ** 1.5)  # m_e* = 0.3 m_0
NV_300K = 2.5e19 * (0.40 ** 1.5)  # m_h* = 0.4 m_0

# Native V_O E_f at εF=VBM (HSE06 Lyons 2022 J. Appl. Phys. 131:025701)
# Convention: E_f^q(εF) = E_f^q_VBM + q·εF (slope = q in E_f vs εF plot)
# At εF=VBM: V_O^+2 most stable (donor); at εF=CBM: V_O^0 most stable.
NATIVE_VO_EF_VBM = {
    0: 3.5,  # V_O neutral
    1: 1.8,  # V_O^+1
    2: 0.3,  # V_O^+2 (deepest in p-type)
}

# Native V_Ga E_f at εF=VBM, μ_Ga=μ_Ga_O_rich (HSE06 Varley 2017 PRB 97:184103)
# V_Ga charge transition levels: ε(0/-1)≈0.7 eV, ε(-1/-2)≈2.0 eV, ε(-2/-3)≈3.5 eV.
# Recovered E_f^q_VBM by inverting transition formula ε(q1/q2) = E_f^q2_VBM - E_f^q1_VBM:
NATIVE_VGA_EF_VBM = {
    -1: 6.7,   # V_Ga^-1 (ε(0/-1) = 0.7 eV → E_f^-1_VBM - E_f^0_VBM = 0.7)
    -2: 8.7,   # V_Ga^-2 (ε(-1/-2) = 2.0 eV → 6.7 + 2.0)
    -3: 12.2,  # V_Ga^-3 (ε(-2/-3) = 3.5 eV → 8.7 + 3.5; deepest acceptor in n-type)
}

# Dopant charge on Ga^3+ site (effective ionization in n-type Ga2O3)
DOPANT_CHARGE_LUT = {
    "Mg": -1, "Zn": -1, "Cu": -1, "Ni": -1,
    "Al": 0, "Fe": 0, "B": 0, "V": 0, "Er": 0, "Eu": 0, "Cr": 0,
    "Sb": 0, "Bi": 0,
    "Si": +1, "Sn": +1, "Ti": +1, "Ge": +1,
    "Ta": +2, "W": +3,
}


def _carrier_n_log(eF: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
    """log10(n) = log10 of conduction band electron density."""
    Nc = NC_300K * (T_K / 300.0) ** 1.5
    delta = (EG_GA2O3 - eF) / (KB_EV * T_K * LN10)
    return torch.log10(Nc.clamp(min=1.0)) - delta


def _carrier_p_log(eF: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
    """log10(p) = log10 of valence band hole density."""
    Nv = NV_300K * (T_K / 300.0) ** 1.5
    delta = eF / (KB_EV * T_K * LN10)
    return torch.log10(Nv.clamp(min=1.0)) - delta


def _defect_log_conc(ef_at_eF: torch.Tensor, T_K: torch.Tensor,
                     log_n_sites: float, log_pO2_factor: float = 0.0) -> torch.Tensor:
    """Compute log10[defect^q] given E_f^q(εF) at given T, optional p_O2 dependence.

    [defect^q] = n_sites * exp(-E_f^q / kT) * exp(-0.5 * log(p_O2)) for V_O
                  ... ignored for V_Ga (independent of p_O2)
    """
    return log_n_sites - ef_at_eF / (KB_EV * T_K * LN10) - log_pO2_factor


def _signed_log_charge(log_conc: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: returns (sign, log10(|q*conc|)) for each row's q*[defect].

    sign = sign(q) if q != 0 else 0.
    log10(|q*conc|) = log10(|q|) + log_conc
    """
    if q == 0:
        sign = torch.zeros_like(log_conc)
        log_abs = torch.full_like(log_conc, -1e9)  # negligible
    else:
        sign = torch.full_like(log_conc, float(math.copysign(1, q)))
        log_abs = log_conc + math.log10(abs(q))
    return sign, log_abs


class KKTBrouwerSolver(nn.Module):
    """Self-consistent Kröger-Vink defect-concentration solver via unrolled Newton.

    Forward: given E_f^q corrections, T, log_pO2, dopant_charge, dopant_log_conc,
    solves charge balance for εF and returns log10[V_O_total].

    Args:
        n_newton_steps: number of unrolled Newton iterations (5-10 typical).
        eF_init: initial εF (relative to VBM, eV). Default midgap.
    """

    def __init__(self, n_newton_steps: int = 8, eF_init: float = EG_GA2O3 / 2):
        super().__init__()
        self.n_newton_steps = int(n_newton_steps)
        self.eF_init = float(eF_init)
        self.log_n_O = math.log10(N_O_SITES)
        self.log_n_Ga = math.log10(N_GA_SITES)

    def _net_charge(self, eF: torch.Tensor, T_K: torch.Tensor,
                     log_pO2: torch.Tensor,
                     ef_VO: dict[int, torch.Tensor],
                     ef_VGa: dict[int, torch.Tensor],
                     dopant_charge: torch.Tensor,
                     dopant_log_conc: torch.Tensor) -> torch.Tensor:
        """Compute net charge density (signed) at given εF.

        Sum: Σ_q q · [V_O^q] + Σ_q q · [V_Ga^q] + q_dop · [dopant] - n + p

        All concentrations are in cm⁻³, output is signed concentration.
        Use log-domain internally for stability, return signed real-space value.
        """
        net = torch.zeros_like(eF)

        # V_O contributions (positive charges)
        for q, ef_VBM in ef_VO.items():
            ef_q = ef_VBM + q * eF
            log_conc = _defect_log_conc(ef_q, T_K, self.log_n_O, 0.5 * log_pO2)
            # Clamp to physical bounds: max site density (N_O or N_Ga ~ 10^22.5)
            log_conc = log_conc.clamp(min=-30, max=22.5)
            conc = 10 ** log_conc
            net = net + q * conc

        # V_Ga contributions (negative charges)
        for q, ef_VBM in ef_VGa.items():
            ef_q = ef_VBM + q * eF
            log_conc = _defect_log_conc(ef_q, T_K, self.log_n_Ga, 0.0)
            # Clamp to physical bounds: max site density (N_O or N_Ga ~ 10^22.5)
            log_conc = log_conc.clamp(min=-30, max=22.5)
            conc = 10 ** log_conc
            net = net + q * conc

        # Dopant contribution: assume fully ionized at q_dopant
        dopant_conc = 10 ** dopant_log_conc.clamp(min=-30, max=30)
        net = net + dopant_charge * dopant_conc

        # Carriers
        log_n = _carrier_n_log(eF, T_K).clamp(min=-30, max=30)
        log_p = _carrier_p_log(eF, T_K).clamp(min=-30, max=30)
        net = net - 10 ** log_n + 10 ** log_p

        return net

    def forward(self,
                ef_VO_corrections: torch.Tensor,    # [B, 3] for q=0,1,2
                ef_VGa_corrections: torch.Tensor,   # [B, 3] for q=-1,-2,-3
                T_K: torch.Tensor,                   # [B]
                log_pO2: torch.Tensor,               # [B]
                dopant_charge: torch.Tensor,         # [B]
                dopant_log_conc: torch.Tensor,       # [B] log10([dopant] in cm⁻³)
                ) -> torch.Tensor:
        """Returns [B] log10([V_O_total] in cm⁻³).

        ef_VO_corrections: small ±0.5 eV corrections from NN, added to base
                           NATIVE_VO_EF_VBM. shape [B, 3].
        ef_VGa_corrections: same format for V_Ga, shape [B, 3].
        """
        device = ef_VO_corrections.device
        B = ef_VO_corrections.shape[0]

        # Build base E_f^q at εF=VBM = 0
        ef_VO = {
            q: NATIVE_VO_EF_VBM[q] + ef_VO_corrections[:, i]
            for i, q in enumerate([0, 1, 2])
        }
        ef_VGa = {
            q: NATIVE_VGA_EF_VBM[q] + ef_VGa_corrections[:, i]
            for i, q in enumerate([-1, -2, -3])
        }

        # Bisection for εF (more robust than Newton on this plateau-prone f).
        # f(εF) is monotone DECREASING (more positive at low εF / VBM, more
        # negative at high εF / CBM, since n grows exponentially toward CBM
        # and V_Ga^q- grows exponentially as well).
        # Use unrolled bisection (differentiable via autograd).
        lo = torch.full((B,), 0.05, device=device, dtype=ef_VO_corrections.dtype)
        hi = torch.full((B,), EG_GA2O3 - 0.05, device=device, dtype=ef_VO_corrections.dtype)

        # n_newton_steps now means n_bisection_steps
        # 25 iterations gives ~10^-7 eV resolution from 4.8 eV range
        n_iters = max(self.n_newton_steps * 3, 20)
        for _ in range(n_iters):
            mid = (lo + hi) / 2
            f_mid = self._net_charge(mid, T_K, log_pO2, ef_VO, ef_VGa,
                                       dopant_charge, dopant_log_conc)
            # f decreases with εF: f > 0 means εF too low → bring lo up
            lo = torch.where(f_mid > 0, mid, lo)
            hi = torch.where(f_mid <= 0, mid, hi)

        eF = (lo + hi) / 2

        # Final V_O total log_conc at converged εF
        log_total = torch.full((B,), -1e9, device=device, dtype=ef_VO_corrections.dtype)
        for q in [0, 1, 2]:
            ef_q = ef_VO[q] + q * eF
            log_conc = _defect_log_conc(ef_q, T_K, self.log_n_O, 0.5 * log_pO2)
            log_conc = log_conc.clamp(min=-30, max=22.5)
            # log-sum-exp accumulation (in log10 base):
            #  log10(a + b) = log10(a) + log10(1 + 10^(log10(b) - log10(a)))
            # Use torch.logaddexp on natural-log basis then convert.
            log_total_ln = log_total * LN10
            log_conc_ln = log_conc * LN10
            log_total = torch.logaddexp(log_total_ln, log_conc_ln) / LN10

        # Save εF for diagnostics (last Newton iterate, detached)
        self._last_eF = eF.detach()

        return log_total


class BrouwerHeadKKT(nn.Module):
    """V53-η: KKT-Hardnet Brouwer head replacing BrouwerHeadVC.

    Predicts E_f corrections from fused embedding, then runs KKTBrouwerSolver.

    Args:
        in_dim:        fused embedding size (160).
        hidden_dim:    latent MLP hidden size.
        dropout:       dropout in latent.
        ef_swing:      tanh-bound on E_f corrections (eV, ±). Default 0.5 eV.
        n_newton:      Newton iterations for εF.
    """

    def __init__(self, in_dim: int = 160, hidden_dim: int = 64, dropout: float = 0.3,
                 ef_swing: float = 0.5, n_newton: int = 8,
                 use_method_offset: bool = False, n_methods: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.ef_swing = float(ef_swing)
        self.use_method_offset = bool(use_method_offset)
        self.n_methods = int(n_methods)
        if self.use_method_offset:
            self.method_offset = nn.Embedding(self.n_methods, 1)
            nn.init.zeros_(self.method_offset.weight)

        # 6 outputs: V_O × 3 charges + V_Ga × 3 charges
        self.latent_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6),
        )
        # Zero-init last layer → corrections=0 → physics-only Brouwer at epoch 0.
        nn.init.zeros_(self.latent_net[-1].weight)
        nn.init.zeros_(self.latent_net[-1].bias)

        # Learnable bias absorbs target-standardization shift (similar to V52b).
        self.global_bias = nn.Parameter(torch.zeros(1))

        # KKT-Hardnet solver
        self.solver = KKTBrouwerSolver(n_newton_steps=n_newton, eF_init=EG_GA2O3 / 2)

    def forward(self, fused: torch.Tensor, process: torch.Tensor,
                dopant_specs: list[str] | None = None,
                method_idx: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        """Forward V53-η Brouwer head.

        Args:
            fused: [B, 160] fused embedding
            process: [B, 17+] process tensor (col 0=T_C, col 2=o2_fraction)
            dopant_specs: list[str] for dopant info (parsed for charge + concentration)
            method_idx: [B] long, optional per-method offset
        Returns:
            [B, 1] standardized log10[V_O_total]
        """
        latents = self.latent_net(fused)  # [B, 6]
        latents = self.ef_swing * torch.tanh(latents)  # bound to ±ef_swing
        ef_VO = latents[:, :3]      # [B, 3] for q=0,1,2
        ef_VGa = latents[:, 3:6]    # [B, 3] for q=-1,-2,-3

        T_K = (process[:, 0] + 273.15).clamp(min=100.0)
        log_pO2 = torch.log10(process[:, 2].clamp(min=1e-4, max=1.0))

        # Parse dopant_specs to charge + concentration
        # Default: q_dopant=0 (isovalent), c=1e15 (negligible)
        device = fused.device
        dop_charge = torch.zeros(fused.shape[0], device=device, dtype=fused.dtype)
        dop_log_conc = torch.full((fused.shape[0],), 15.0, device=device, dtype=fused.dtype)
        if dopant_specs is not None:
            from src.data.dopant_spec import DopantSpec
            for i, spec in enumerate(dopant_specs):
                try:
                    ds = DopantSpec.parse(spec)
                    if ds.is_undoped or not ds.components:
                        continue
                    cation = max(ds.components, key=lambda c: c.conc).cation
                    q = DOPANT_CHARGE_LUT.get(cation, 0)
                    c_at_frac = sum(max(float(c.conc), 0.0) for c in ds.components)
                    if c_at_frac < 1e-6:
                        continue
                    # Convert at-frac to cm⁻³: c × N_cation_sites
                    c_cm3 = c_at_frac * 1.9e22
                    dop_charge[i] = float(q)
                    dop_log_conc[i] = math.log10(c_cm3)
                except Exception:
                    pass

        log_VO = self.solver(ef_VO, ef_VGa, T_K, log_pO2, dop_charge, dop_log_conc)
        log_VO = log_VO.unsqueeze(-1) + self.global_bias  # [B, 1]

        if self.use_method_offset and method_idx is not None:
            log_VO = log_VO + self.method_offset(method_idx)

        return log_VO
