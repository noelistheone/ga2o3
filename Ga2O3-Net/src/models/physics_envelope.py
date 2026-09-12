"""
Phase 24B — physics envelopes for PDR and VC head outputs.

Two interpretable, parameter-light modules that wrap the raw head output
with a physics-direction prior:

* :class:`PhysicsEnvelopePDR` — multiplicative envelope built from
    - **bandgap cliff**: sigmoid in (E_photon − E_g) → photoresponse vanishes
      below the bandgap.
    - **temperature bell**: gaussian around a learnable T_opt (anneal sweet
      spot — too cold → frozen defects; too hot → re-oxidation).

  ``pred_final = (pred_raw - floor) * env + floor``

  When ``env → 1`` the prediction is data-driven; when ``env → 0`` the
  output collapses to a learnable ``floor`` (standardized "no-response"
  baseline, expected to converge near (log_PDR=0 − μ)/σ).

* :class:`PhysicsEnvelopeVC` — additive correction from process channels
  not represented in V_O thermodynamics: T (Boltzmann generation, +) and
  pO2 proxy (oxidation, −). Trainable rates `α_T`, `α_O2` are constrained
  non-negative via softplus so the physics direction cannot flip.

All envelope parameters are *interpretable physics constants* that can be
read out after training (T_opt, σ_T, σ_λ, α_T, α_O2) and compared with
literature values.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.physics_features import (
    BASE_E_G_EV,
    PHYSICS_FEATURE_NAMES,
)


def _idx(name: str) -> int:
    return PHYSICS_FEATURE_NAMES.index(name)


def _inv_softplus(y: float) -> float:
    """Return z such that softplus(z) = y, for y > 0."""
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


_E_PHOTON_MINUS_EG_IDX = _idx("E_photon_minus_Eg")
_KT_IDX = _idx("kT_eV")
_PO2_IDX = _idx("pO2_proxy")


class PhysicsEnvelopePDR(nn.Module):
    """
    Photo-dark ratio envelope.

    Args:
        target_idx: column index in the head output corresponding to PDR.
        T_opt_init: initial annealing optimum (°C). Common range 500–800 °C.
        sigma_T_init: initial bell width (°C).
        delta_eV_init: initial bandgap soft margin (eV).
        sigma_lambda_init: initial bandgap transition width (eV).
        floor_init: initial standardized "no-response" baseline. -1.0 is a
                    rough match for log_PDR=0 after typical standardization.
    """

    def __init__(
        self,
        target_idx: int = 0,
        T_opt_init: float = 700.0,
        sigma_T_init: float = 250.0,
        delta_eV_init: float = 0.05,
        sigma_lambda_init: float = 0.15,
        floor_init: float = -1.0,
        apply_to_output: bool = True,
    ):
        super().__init__()
        self.target_idx = target_idx
        # Phase 25-soft: when False, .forward() is a no-op so the model output
        # stays free; the trainer reads `.envelope(...)` directly to compute a
        # hinge loss in physical space. Envelope params remain trainable.
        self.apply_to_output = apply_to_output

        # T_opt is a free parameter in °C; bound to [300, 1100] via sigmoid
        self._T_opt_logit = nn.Parameter(
            torch.tensor(self._inv_sigmoid_bounded(T_opt_init, 300.0, 1100.0))
        )
        self._sigma_T_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sigma_T_init - 50.0))
        )
        self._delta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(delta_eV_init))
        )
        self._sigma_l_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sigma_lambda_init))
        )
        self.floor = nn.Parameter(torch.tensor(float(floor_init)))

    # ── Param transforms (interpretable read-out) ──────────────────────────
    @property
    def T_opt(self) -> torch.Tensor:
        return self._sigmoid_bounded(self._T_opt_logit, 300.0, 1100.0)

    @property
    def sigma_T(self) -> torch.Tensor:
        return F.softplus(self._sigma_T_raw) + 50.0

    @property
    def delta_eV(self) -> torch.Tensor:
        return F.softplus(self._delta_raw) + 1e-3

    @property
    def sigma_lambda(self) -> torch.Tensor:
        return F.softplus(self._sigma_l_raw) + 1e-3

    @staticmethod
    def _inv_sigmoid_bounded(x: float, lo: float, hi: float) -> float:
        u = (x - lo) / (hi - lo)
        u = max(min(u, 0.999), 0.001)
        return math.log(u / (1.0 - u))

    @staticmethod
    def _sigmoid_bounded(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(z)

    # ── Envelope ───────────────────────────────────────────────────────────
    def envelope(
        self,
        process: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns envelope ∈ [0, 1] of shape [B].

        env = sigmoid((E_photon - E_g + δ) / σ_λ)
              · exp(-(T - T_opt)^2 / (2 σ_T^2))
        """
        T_C = process[:, 0]
        e_phot_minus_eg = physics[:, _E_PHOTON_MINUS_EG_IDX]

        env_lambda = torch.sigmoid((e_phot_minus_eg + self.delta_eV) / self.sigma_lambda)
        env_T = torch.exp(-((T_C - self.T_opt) ** 2) / (2.0 * self.sigma_T ** 2))
        return env_lambda * env_T

    def forward(
        self,
        out: torch.Tensor,
        process: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply envelope to a single column of `out` ([B, num_targets]).

        pred_final = (pred_raw − floor) · env + floor

        No-op when ``apply_to_output=False`` (Phase 25-soft hinge mode).
        """
        if not self.apply_to_output:
            return out

        env = self.envelope(process, physics)            # [B]

        if out.dim() != 2:
            raise ValueError(f"out must be [B, T], got {tuple(out.shape)}")

        target_col = out[:, self.target_idx]
        new_col = (target_col - self.floor) * env + self.floor
        out_new = out.clone()
        out_new[:, self.target_idx] = new_col
        return out_new

    def extra_repr(self) -> str:
        return (
            f"target_idx={self.target_idx}, T_opt={self.T_opt.item():.1f}°C, "
            f"σ_T={self.sigma_T.item():.1f}°C, "
            f"δ={self.delta_eV.item():.3f}eV, σ_λ={self.sigma_lambda.item():.3f}eV, "
            f"floor={self.floor.item():+.2f}"
        )


class PhysicsEnvelopeVC(nn.Module):
    """
    Vacancy-concentration envelope (additive).

    pred_VC_final = pred_VC_raw + α_T · (T_C/1000 − T_ref) − α_O2 · pO2_proxy

    Both α_T and α_O2 are kept non-negative (softplus), preserving the
    physics direction:
      * higher T (relative to reference) → more V_O
      * higher μ_O (oxidising atmo) → fewer V_O

    Args:
        target_idx: column index of vacancy_concentration in head output.
        alpha_T_init, alpha_O2_init: initial rate magnitudes (in standardized
            log_VC units per unit normalized predictor). Kept small to avoid
            dominating data signal.
        T_ref: reference temperature in units of T/1000 (°C). Default 0.7
            (700°C) — a typical mid-anneal baseline for β-Ga₂O₃.
    """

    def __init__(
        self,
        target_idx: int = 1,
        alpha_T_init: float = 0.10,
        alpha_O2_init: float = 0.20,
        T_ref: float = 0.70,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.T_ref = float(T_ref)
        self._alpha_T_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_T_init)))
        self._alpha_O2_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_O2_init)))

    @property
    def alpha_T(self) -> torch.Tensor:
        return F.softplus(self._alpha_T_raw)

    @property
    def alpha_O2(self) -> torch.Tensor:
        return F.softplus(self._alpha_O2_raw)

    def forward(
        self,
        out: torch.Tensor,
        process: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        T_norm = process[:, 0] / 1000.0
        pO2 = physics[:, _PO2_IDX]
        delta = self.alpha_T * (T_norm - self.T_ref) - self.alpha_O2 * pO2  # [B]

        if out.dim() != 2:
            raise ValueError(f"out must be [B, T], got {tuple(out.shape)}")
        out_new = out.clone()
        out_new[:, self.target_idx] = out[:, self.target_idx] + delta
        return out_new

    def extra_repr(self) -> str:
        return (
            f"target_idx={self.target_idx}, "
            f"α_T={self.alpha_T.item():.3f}, α_O2={self.alpha_O2.item():.3f}, "
            f"T_ref={self.T_ref:.2f}·1000°C"
        )
