"""
Composition Stream: extract XenonPy chemical features for doped Ga₂O₃,
then map them to a fixed-dim vector via a small MLP.

Accepts arbitrary dopant specifications (single element, multi-element,
compound, multi-compound) via DopantSpec strings.

Pipeline:
  dopant_spec_str → DopantSpec → xenonpy_formula
    → XenonPy Compositions → raw feature vector
    → StandardScaler
    → Linear → ReLU → Linear → 64-dim output
"""

from __future__ import annotations

import pickle
import logging
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data.dopant_spec import DopantSpec, parse_spec

logger = logging.getLogger(__name__)

_xenonpy_calculator = None


def _get_calculator():
    global _xenonpy_calculator
    if _xenonpy_calculator is None:
        try:
            from xenonpy.descriptor import Compositions
            _xenonpy_calculator = Compositions()
        except ImportError:
            raise ImportError("Install XenonPy: pip install xenonpy")
    return _xenonpy_calculator


@lru_cache(maxsize=2048)
def _raw_xenonpy_features(xenonpy_formula: str) -> np.ndarray:
    """Cached XenonPy feature extraction for a given formula string."""
    calc = _get_calculator()
    result = calc.transform([xenonpy_formula])
    arr = result.values[0]
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.astype(np.float32)


def spec_to_xenonpy_features(spec_str: str, total_conc_frac: float | None = None) -> np.ndarray:
    """
    Convert a dopant spec string to a XenonPy feature vector.

    Args:
        spec_str       : DopantSpec string, e.g. "Fe:0.0265" or "SnO2:0.013,MgO:0.013".
        total_conc_frac: Total concentration fraction; used when spec_str lacks
                         per-component concentrations.

    Returns:
        np.ndarray of shape [raw_dim] (typically ~290).
    """
    spec = parse_spec(spec_str, total_conc_frac)
    formula = spec.to_xenonpy_formula()
    return _raw_xenonpy_features(formula)


class CompositionStream(nn.Module):
    """
    Maps a dopant specification → 64-dim composition embedding.

    Accepts arbitrary dopant specs (single/multi-element, compound, multi-compound).

    Args:
        raw_dim   : XenonPy output dimension (detected on first call; default 290).
        hidden_dim: MLP hidden dimension.
        out_dim   : Output embedding dimension.
        dropout   : Dropout rate.
    """

    def __init__(
        self,
        raw_dim: int = 290,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_dim = raw_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._scaler: StandardScaler | None = None

    # ── Scaler management ─────────────────────────────────────────────────────

    def fit_scaler(self, dopant_specs: list[str], total_concs: list[float] | None = None):
        """
        Fit the StandardScaler on a list of dopant spec strings.

        Args:
            dopant_specs: list of spec strings, one per training sample.
            total_concs : optional list of total concentration fractions,
                          used when specs lack embedded concentrations.
        """
        if total_concs is None:
            total_concs = [None] * len(dopant_specs)

        raw_features = []
        for spec_str, tc in zip(dopant_specs, total_concs):
            raw_features.append(spec_to_xenonpy_features(spec_str, tc))

        X = np.stack(raw_features)

        # Auto-detect actual dim from XenonPy output
        if X.shape[1] != self.raw_dim:
            logger.info(
                f"XenonPy returned {X.shape[1]}-dim features; "
                f"updating model raw_dim from {self.raw_dim} → {X.shape[1]}"
            )
            self.raw_dim = X.shape[1]
            self.mlp[0] = nn.Linear(X.shape[1], self.mlp[0].out_features)

        self._scaler = StandardScaler().fit(X)
        logger.info(f"CompositionStream scaler fitted on {len(dopant_specs)} samples "
                    f"(raw_dim={self.raw_dim})")

    def save_scaler(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "raw_dim": self.raw_dim}, f)

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            self._scaler = obj["scaler"]
            if obj.get("raw_dim") and obj["raw_dim"] != self.raw_dim:
                self.raw_dim = obj["raw_dim"]
                self.mlp[0] = nn.Linear(self.raw_dim, self.mlp[0].out_features)
        else:
            self._scaler = obj  # legacy: plain scaler object

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        dopant_specs: list[str],
        total_concs: list[float] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            dopant_specs: list of spec strings, length B.
                          e.g. ["Fe:0.0265", "SnO2:0.013,MgO:0.013", …]
            total_concs : optional list of total concentration fractions,
                          used when specs lack per-component concentrations.

        Returns:
            Tensor [B, out_dim]
        """
        if total_concs is None:
            total_concs = [None] * len(dopant_specs)

        raw_list = [
            spec_to_xenonpy_features(s, tc)
            for s, tc in zip(dopant_specs, total_concs)
        ]
        raw = np.stack(raw_list)  # [B, raw_dim]

        # Auto-adapt if XenonPy dim changed (e.g. different version)
        if raw.shape[1] != self.mlp[0].in_features:
            raise ValueError(
                f"XenonPy returned {raw.shape[1]}-dim features but MLP expects "
                f"{self.mlp[0].in_features}. Re-fit the scaler or rebuild the model."
            )

        if self._scaler is not None:
            raw = self._scaler.transform(raw)

        x = torch.tensor(raw, dtype=torch.float32,
                          device=next(self.parameters()).device)
        return self.mlp(x)

    def get_scaled_features(
        self,
        dopant_specs: list[str],
        total_concs: list[float] | None = None,
    ) -> torch.Tensor | None:
        """
        Return scaled XenonPy features [B, raw_dim] without passing through MLP.

        Used as reconstruction target for the composition decoder auxiliary loss.
        Returns None if the scaler has not been fitted yet.

        Args:
            dopant_specs: list of spec strings, length B.
            total_concs : optional per-sample total concentration fractions.

        Returns:
            Tensor [B, raw_dim] of scaled features, or None.
        """
        if self._scaler is None:
            return None
        if total_concs is None:
            total_concs = [None] * len(dopant_specs)

        raw_list = [
            spec_to_xenonpy_features(s, tc)
            for s, tc in zip(dopant_specs, total_concs)
        ]
        raw = np.stack(raw_list)  # [B, raw_dim]

        if raw.shape[1] != self.raw_dim:
            return None  # dim mismatch, skip reconstruction safely

        scaled = self._scaler.transform(raw)
        return torch.tensor(
            scaled, dtype=torch.float32, device=next(self.parameters()).device
        )
