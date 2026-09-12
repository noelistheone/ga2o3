"""
Process Stream: encode annealing conditions + fabrication method into a k-dimensional
feature vector.

Architecture
============
The 18-dim input tensor (Phase 5C layout, produced by build_process_tensor /
Ga2O3ExpDataset) is split into two branches:

  Continuous branch (indices 0–11):
    [temperature_C, time_min, o2_fraction,
     is_plasma_enhanced, has_nitrogen_species,
     has_plasma_dep, has_anneal, has_buffer, on_foreign_substrate,
     measurement_voltage_V, measurement_wavelength_nm,
     log10_concentration]
    → StandardScaler (fitted on training fold only, leakage-free)
    → normalized 12-dim vector

  Method branch (indices 12–16, one-hot bits):
    [method_sputtering, method_pld, method_cvd_ald, method_wet, method_evaporation]
    → converted to integer index (argmax; all-zeros → index 5 = "other")
    → nn.Embedding(n_methods=6, method_embed_dim=3)
    → 3-dim learned method embedding

  [normalized_11dim ‖ method_emb_3dim] → Linear(14, hidden_dim) → ReLU → Dropout
  → Linear(hidden_dim, out_dim=32)

Method index mapping
--------------------
  0 = sputtering   (RF/DC magnetron, reactive sputtering, PVD)
  1 = PLD          (pulsed laser deposition, laser ablation)
  2 = CVD/ALD      (CVD, MOCVD, ALD, PECVD, LPCVD, MBE)
  3 = wet          (hydrothermal, sol-gel, CBD, spray, electrodeposition)
  4 = evaporation  (thermal / e-beam evaporation)
  5 = other        (all method bits = 0; unknown or mixed)

Rationale for embedding over one-hot
-------------------------------------
One-hot treats all methods as equidistant. The learned embedding can capture
physical proximity: PLD and sputtering are both vacuum-based (similar film
densities, defect formation channels) while wet chemistry is orthogonal.
With ≤127 samples and some methods having only 5–10 samples, a 3-dim
embedding (rule of thumb: min(50, ⌈n_categories/2⌉) = 3) substantially
reduces overfitting risk vs. 5 independent one-hot weight columns, while
allowing the model to discover structure the one-hot encoding cannot represent.

Reference: Guo & Berkhahn, "Entity Embeddings of Categorical Variables",
arXiv:1604.06737 (2016).

Literature context on method–temperature correlation
------------------------------------------------------
The fabrication method and annealing temperature are significantly correlated
in the Ga₂O₃ thin-film literature:
  - PLD:   500–800°C (cracking / Al-diffusion risk above 800°C on sapphire)
  - MOCVD: 900–1100°C (κ→β phase conversion requires high T)
  - PEALD: often no post-anneal, or 200–500°C (crystalline β as-grown)
  - Sputtering: 600–1000°C (widest range, most common)
  - Wet/sol-gel: 700–1100°C
One-hot encoding in a Linear(10, 64) layer cannot capture this joint structure.
The embedding allows the model to learn "PLD at 700°C is typical; sputtering
at 700°C is also typical; MOCVD at 700°C is atypically low" from data.

Use build_process_tensor() from src.data.experimental_dataset to construct the
10-dim input tensor from human-readable atmosphere and method strings.
"""

import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Phase 5C: indices in the 18-dim process tensor
_CONTINUOUS_DIM = 12          # indices 0–11 are continuous, normalised by StandardScaler
_METHOD_BITS = 5              # indices 12–16 are one-hot method bits
_N_METHODS_DEFAULT = 6        # 5 named methods + 1 unknown/other
_N_SUBSTRATES_DEFAULT = 7     # sapphire, Si, GaN, MgO, oxide_template, native_bulk, other


class ProcessStream(nn.Module):
    """
    Process condition encoder with learned method and substrate embeddings.

    Input: 18-dim process tensor [B, 18] (Phase 5C)
      - Indices 0–11:  continuous (T, t, O₂, atm-plasma, N, plasma-dep, anneal,
        buffer, foreign-sub, meas-V, meas-λ, log10_concentration) → StandardScaler
        → normalised 12-dim
      - Indices 12–16: one-hot method bits → integer index → Embedding(n_methods, embed_dim)
      - Index 17:      substrate category index (0–6) → Embedding(n_substrates, sub_embed_dim)
    Output: [B, out_dim]  (default 32)

    Args:
        continuous_dim:      Number of continuous scalars to normalise (default 12).
        method_embed_dim:    Embedding dimension for fabrication method (default 3).
        n_methods:           Number of method categories incl. 'other' (default 6).
        substrate_embed_dim: Embedding dimension for substrate category (default 3).
        n_substrates:        Number of substrate categories incl. 'other' (default 7).
        hidden_dim:          MLP hidden layer size (default 64).
        out_dim:             Output embedding dimension (default 32).
        dropout:             Dropout rate (default 0.2).
        in_dim:              Total input width (default 18). Stored for external
                             queries; not used inside the MLP.
    """

    def __init__(
        self,
        continuous_dim: int = _CONTINUOUS_DIM,
        method_embed_dim: int = 3,
        n_methods: int = _N_METHODS_DEFAULT,
        substrate_embed_dim: int = 3,
        n_substrates: int = _N_SUBSTRATES_DEFAULT,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.2,
        in_dim: int = 18,
    ):
        super().__init__()
        self.continuous_dim      = continuous_dim
        self.method_embed_dim    = method_embed_dim
        self.n_methods           = n_methods
        self.substrate_embed_dim = substrate_embed_dim
        self.n_substrates        = n_substrates
        self.out_dim             = out_dim
        self.in_dim              = in_dim

        # Learnable method embedding: captures physical proximity between methods
        self.method_embedding    = nn.Embedding(n_methods, method_embed_dim)
        # Learnable substrate embedding: captures interface/mismatch similarities
        # (e.g. sapphire ≈ oxide_template more than Si ≈ GaN).
        self.substrate_embedding = nn.Embedding(n_substrates, substrate_embed_dim)

        # MLP input: normalised continuous dims + method embedding + substrate embedding
        mlp_in = continuous_dim + method_embed_dim + substrate_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self._scaler: StandardScaler | None = None

    # ── Scaler ────────────────────────────────────────────────────────────────

    def fit_scaler(self, process_array: np.ndarray):
        """
        Fit StandardScaler on the *continuous* portion of the process array.

        Only the first ``continuous_dim`` columns are used (indices 0–4).
        The one-hot method bits (indices 5–9) are converted to a categorical
        index and embedded — they are never passed through the scaler.

        Args:
            process_array: np.ndarray of shape [N, in_dim] (full 10-dim array
                           from Ga2O3ExpDataset.get_process_array()).
        """
        self._scaler = StandardScaler()
        self._scaler.fit(process_array[:, : self.continuous_dim])
        logger.info(
            f"ProcessStream scaler fitted on {self.continuous_dim} continuous dims — "
            f"mean={self._scaler.mean_.round(2)}, scale={self._scaler.scale_.round(2)}"
        )

    def save_scaler(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._scaler, f)

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            self._scaler = pickle.load(f)

    def normalize(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Apply StandardScaler to [B, continuous_dim] continuous features.

        Returns input unchanged if the scaler has not been fitted yet
        (e.g., during a forward pass before training; the trainer always
        calls fit_scaler before the first epoch).
        """
        if self._scaler is None:
            return continuous
        arr = continuous.cpu().numpy()
        arr_norm = self._scaler.transform(arr).astype(np.float32)
        return torch.tensor(arr_norm, device=continuous.device)

    # ── Method index extraction ────────────────────────────────────────────────

    def _method_bits_to_idx(self, process: torch.Tensor) -> torch.Tensor:
        """
        Convert the 5 one-hot method bits to a LongTensor index per sample.

        Slice is ``process[:, continuous_dim : continuous_dim + 5]`` so that the
        substrate index at position ``continuous_dim + 5`` (Phase 2E) is NOT
        included in the argmax window.

        Bit layout:
          bit 0 → index 0 (sputtering)
          bit 1 → index 1 (PLD)
          bit 2 → index 2 (CVD/ALD)
          bit 3 → index 3 (wet)
          bit 4 → index 4 (evaporation)
          all zeros → index n_methods-1 (other/unknown)

        Multiple active bits are resolved by argmax (first/highest-value wins).
        """
        start = self.continuous_dim
        bits = process[:, start : start + _METHOD_BITS]   # [B, 5]
        idx  = bits.long().argmax(dim=1)
        any_active = bits.sum(dim=1) > 0
        unknown = torch.full_like(idx, self.n_methods - 1)
        return torch.where(any_active, idx, unknown)

    def _substrate_idx(self, process: torch.Tensor) -> torch.Tensor:
        """Extract the substrate category index at position continuous_dim+5 (Phase 2E)."""
        raw = process[:, self.continuous_dim + _METHOD_BITS]   # [B]  float
        idx = raw.long().clamp(min=0, max=self.n_substrates - 1)
        return idx

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, process: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of process condition tensors.

        Args:
            process: Tensor [B, in_dim] (default 17-dim, Phase 2E).

        Returns:
            Tensor [B, out_dim]
        """
        # ── Continuous branch ─────────────────────────────────────────────────
        continuous = process[:, : self.continuous_dim]   # [B, 12]
        x_cont     = self.normalize(continuous)

        # ── Method embedding branch ───────────────────────────────────────────
        method_idx = self._method_bits_to_idx(process)
        x_meth     = self.method_embedding(method_idx)   # [B, method_embed_dim]

        # ── Substrate embedding branch (Phase 2E) ─────────────────────────────
        sub_idx    = self._substrate_idx(process)
        x_sub      = self.substrate_embedding(sub_idx)   # [B, substrate_embed_dim]

        x = torch.cat([x_cont, x_meth, x_sub], dim=-1)
        return self.mlp(x)
