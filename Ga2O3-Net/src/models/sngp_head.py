"""Phase 54 V54-A2 — Spectral-Normalized Gaussian Process (SNGP) head.

Liu et al., "Simple and Principled Uncertainty Estimation with Deterministic
Deep Learning via Distance Awareness" arXiv:2006.10108. Regression form
(Liu & Padhy 2022) used here.

Key design:
  - Hidden residual layers wrapped with `spectral_norm` (Lipschitz ≤ 1)
    preserve distance information in the fused embedding space.
  - Output replaced by a *Random Fourier Features GP* (RFF GP): the network
    learns a deterministic non-linear projection that approximates a GP, and
    posterior variance is computed in closed form from a frozen precision
    matrix accumulated during training.
  - Verbatim hyperparameter recommendations from Wei et al.
    medRxiv 2025.02.26.25322983 §2.3:  D=512 RFF dims, ridge λ=1e-4,
    EMA discount γ=0.999, with normal-init RFF weights frozen.

This module is a **drop-in head** consuming the fused embedding [B, in_dim]
and returning either (mean) at training or (mean, var) at inference. It is
agnostic to the BrouwerHeadVC physics math — to integrate with the
analytical Brouwer pipeline, call SNGPHead.forward(fused) to obtain a
calibrated residual on top of BrouwerHeadVC.forward(fused, process), and
add them. The variance is purely the GP variance over the residual.

Rule 1 (soft only): the BrouwerHeadVC analytical output is untouched; this
adds an auxiliary residual head with distance-aware uncertainty.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wrap_spectral_norm(linear: nn.Linear, n_power_iterations: int = 1) -> nn.Linear:
    """Apply spectral normalization via the new parametrize API."""
    try:
        from torch.nn.utils.parametrizations import spectral_norm
        return spectral_norm(linear, n_power_iterations=n_power_iterations)
    except ImportError:
        # Fallback for older torch
        return nn.utils.spectral_norm(linear, n_power_iterations=n_power_iterations)


class SpectralResidualMLP(nn.Module):
    """3-layer residual MLP with spectral-normalized Linear layers.

    Preserves bi-Lipschitz property of the input → hidden mapping, which
    SNGP requires for distance-aware uncertainty (Liu §3.2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = _wrap_spectral_norm(nn.Linear(in_dim, hidden_dim))
        self.layers = nn.ModuleList([
            nn.Sequential(
                _wrap_spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                _wrap_spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            )
            for _ in range(n_layers)
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        h = self.dropout(h)
        for layer in self.layers:
            h = self.act(h + layer(h))
            h = self.dropout(h)
        return h


class RFFGPRegressor(nn.Module):
    """Random Fourier Features approximation to a GP regressor.

    Posterior variance:
        var(x) = (1/scale) * φ(x)^T Σ^{-1} φ(x)
    where Σ = ridge*I + Σ_t φ(x_t) φ(x_t)^T  is accumulated during training
    (frozen at eval time), φ are sinusoidal Random Fourier Features with
    frozen Gaussian-init weights, and `scale` is an EMA-discounted running
    estimate of the noise variance.

    Mean is a single Linear layer over φ(x), trained jointly with the upstream
    SpectralResidualMLP via standard MSE / cross-entropy loss.
    """

    def __init__(self, in_dim: int, rff_dim: int = 512,
                 length_scale: float = 1.0, ridge: float = 1e-4,
                 momentum: float = 0.999):
        super().__init__()
        self.in_dim = in_dim
        self.rff_dim = rff_dim
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        self.momentum = float(momentum)

        # Frozen Random Fourier Features: φ(x) = sqrt(2/D) cos(W x / ℓ + b)
        # W ~ N(0, 1), b ~ U(0, 2π) — both buffers (no gradient).
        self.register_buffer("W_rff", torch.randn(rff_dim, in_dim))
        self.register_buffer("b_rff", torch.rand(rff_dim) * 2.0 * math.pi)
        # Mean head: trainable linear over φ(x)
        self.beta = nn.Linear(rff_dim, 1, bias=False)
        nn.init.zeros_(self.beta.weight)
        # Precision matrix Σ (registered as buffer; updated in train mode only).
        self.register_buffer("precision_matrix",
                              ridge * torch.eye(rff_dim))
        # Posterior covariance cache (computed lazily at eval time after Σ is frozen)
        self.register_buffer("cov_cache",
                              torch.eye(rff_dim) / max(ridge, 1e-12))
        self.cov_dirty = True

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """RFF projection: x ∈ [B, in_dim] → φ(x) ∈ [B, rff_dim]."""
        z = F.linear(x / self.length_scale, self.W_rff, bias=self.b_rff)
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(z)

    def reset_precision(self) -> None:
        """Call once per epoch (or per training run) before accumulating Σ."""
        with torch.no_grad():
            self.precision_matrix.copy_(
                self.ridge * torch.eye(self.rff_dim,
                                       device=self.precision_matrix.device,
                                       dtype=self.precision_matrix.dtype)
            )
        self.cov_dirty = True

    def accumulate_precision(self, x: torch.Tensor) -> None:
        """Online EMA update: Σ ← γ·Σ + (1-γ)·φ(x)^T φ(x) per minibatch."""
        if not self.training:
            return
        with torch.no_grad():
            phi = self._phi(x).detach()                           # [B, D]
            cov_update = phi.T @ phi                              # [D, D]
            self.precision_matrix.mul_(self.momentum).add_(
                cov_update, alpha=(1.0 - self.momentum)
            )
        self.cov_dirty = True

    def _refresh_cov(self) -> None:
        with torch.no_grad():
            # Use Cholesky for stable inversion
            try:
                L = torch.linalg.cholesky(self.precision_matrix)
                inv = torch.cholesky_inverse(L)
            except Exception:
                inv = torch.linalg.pinv(self.precision_matrix)
            self.cov_cache.copy_(inv)
        self.cov_dirty = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, var). Mean is gradient-bearing; var is detached."""
        phi = self._phi(x)                                        # [B, D]
        mean = self.beta(phi).squeeze(-1)                         # [B]
        # Accumulate precision during training (no gradient through Σ).
        if self.training:
            self.accumulate_precision(x)
        # Variance computation needs Σ^{-1} φ — refresh lazily at eval.
        if not self.training:
            if self.cov_dirty:
                self._refresh_cov()
            with torch.no_grad():
                # var(x) = φ^T Σ^{-1} φ
                v = torch.einsum("bi,ij,bj->b", phi, self.cov_cache, phi)
                # Floor at small positive for numerical safety
                var = v.clamp(min=1e-8)
        else:
            var = torch.zeros_like(mean)
        return mean, var


class SNGPHead(nn.Module):
    """Composed: SpectralResidualMLP → RFFGPRegressor.

    Drop-in for any [B, in_dim] → [B, 1] head. Use at training:
        mean, _ = head(fused)
        loss = mse(mean, target_norm) + ...
    Use at inference:
        mean, var = head(fused)
        # var is the calibrated posterior variance (distance-aware)

    To integrate with BrouwerHeadVC, treat SNGPHead output as a residual:
        log_vo_pred = brouwer(fused, process) + sngp(fused).mean
    The variance from SNGP is then the predictive uncertainty on the residual.
    """

    def __init__(self, in_dim: int = 160, hidden_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.1,
                 rff_dim: int = 512, ridge: float = 1e-4,
                 momentum: float = 0.999,
                 length_scale: float | None = None):
        super().__init__()
        self.spec_mlp = SpectralResidualMLP(in_dim, hidden_dim, n_layers, dropout)
        if length_scale is None:
            length_scale = math.sqrt(float(hidden_dim))
        self.gp = RFFGPRegressor(hidden_dim, rff_dim=rff_dim,
                                  length_scale=length_scale, ridge=ridge,
                                  momentum=momentum)

    def reset_precision(self) -> None:
        self.gp.reset_precision()

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.spec_mlp(fused)
        return self.gp(h)
