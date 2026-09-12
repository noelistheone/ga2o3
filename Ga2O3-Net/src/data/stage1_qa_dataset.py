"""
Stage-1 Physics-LLM QA dataset (Phase 58, design doc §5.1.3 / §3.6.3).

Wraps `data/processed/stage1_qa_pairs.jsonl` (and the held-out
`stage1_qa_pairs_val.jsonl`).  Each line is a JSON dict:

    {
      "input":  {dopant, dopant_concentration, atmosphere,
                 temperature_C, fermi_level_eV},
      "label":  {log_vo_predicted, dominant_charge_state, log_K_eq,
                 delta_E_f_eV, reasoning},
      "_source": "kroger|dft_cache|tier1c|literature"
    }

The dataset builds a Qwen2.5 chat prompt from `input` plus the reasoning text
from `label`, tokenizes the concatenation into a SINGLE causal-LM sequence,
and masks the prompt portion of `labels` with -100 so that only the reasoning
tokens contribute to LM loss.

Numeric targets (`log_vo_predicted`, `log_K_eq`, `delta_E_f_eV`) and the
3-way `dominant_charge_state` target are returned per-sample for the numeric
prior head (§3.6.4).  Numerics are STANDARDIZED (z = (x - mean) / std) using
train-only statistics that the caller computes via `compute_numeric_stats`
on the train file and then passes into each dataset instance via
`set_numeric_stats`. The trainer un-standardizes for the MAE metric.

§3.6.1 max_seq_len = 2048 (typical input ~500 tokens — see doc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


# ---- prompt construction (design doc §3.6.3) --------------------------------

_VALENCE_TABLE: dict[str, tuple[str, str]] = {
    # element -> (valence, category)
    "Mg": ("+2", "acceptor"),
    "Zn": ("+2", "acceptor"),
    "Fe": ("+3", "acceptor"),
    "Cu": ("+2", "acceptor"),
    "B":  ("+3", "acceptor"),
    "Si": ("+4", "donor"),
    "Sn": ("+4", "donor"),
    "Ge": ("+4", "donor"),
    "Ti": ("+4", "donor"),
    "Ta": ("+5", "donor"),
    "W":  ("+6", "donor"),
    "V":  ("+5", "donor"),
    "Nb": ("+5", "donor"),
    "Sb": ("+5", "super-donor"),
    "Cr": ("+3", "isovalent"),
    "In": ("+3", "isovalent"),
    "Bi": ("+3", "isovalent"),
    "La": ("+3", "isovalent"),
    "Eu": ("+3", "isovalent"),
    "Er": ("+3", "isovalent"),
    "F":  ("-1", "anion"),
    "N":  ("-3", "anion"),
    "H":  ("+1", "donor"),
    # fallback for undoped
    "undoped": ("0", "isovalent"),
}


def _valence_for(dopant: str) -> tuple[str, str]:
    return _VALENCE_TABLE.get(dopant, ("?", "unknown"))


def build_prompt(inp: dict) -> str:
    """Render the §3.6.3 prompt template (deterministic, no chat tags here —
    the chat template is applied later via the tokenizer)."""
    valence, category = _valence_for(inp["dopant"])
    return (
        "You are an expert in beta-Ga2O3 defect chemistry. Given the following "
        "doping condition, reason about the equilibrium oxygen vacancy "
        "concentration and charge state distribution using Brouwer diagram "
        "thermodynamics.\n\n"
        f"Dopant: {inp['dopant']} (valence: {valence}, category: {category})\n"
        f"Dopant concentration: {float(inp['dopant_concentration']):.6f} (atomic fraction)\n"
        f"Atmosphere: {inp['atmosphere']}\n"
        f"Temperature: {float(inp['temperature_C']):.1f} C\n"
        f"Fermi level estimate: {float(inp['fermi_level_eV']):.2f} eV from VBM\n\n"
        "Reason step by step:\n"
        "1. Identify dopant's substitutional site (Ga_I tetrahedral vs Ga_II octahedral preference)\n"
        "2. Compute mu_O at given atmosphere and temperature via Reuter-Scheffler\n"
        "3. Identify dominant charge state of V_O (0/+1/+2) at this Fermi level\n"
        "4. Apply charge neutrality\n"
        "5. Predict log10[V_O] in cm^-3 units\n\n"
        "Then output:\n"
        "PREDICTED_LOG_VO: <float>\n"
        "DOMINANT_CHARGE: <int>\n"
        "REASONING_CONFIDENCE: <low/medium/high>"
    )


def build_completion(label: dict) -> str:
    """The 'answer' string that the LM is supervised on (loss applied here)."""
    reasoning = (label.get("reasoning") or "").strip()
    log_vo = float(label["log_vo_predicted"])
    dc = int(label["dominant_charge_state"])
    confidence = "high" if abs(log_vo - 5.0) > 1e-6 else "medium"
    return (
        f"{reasoning}\n\n"
        f"PREDICTED_LOG_VO: {log_vo:.2f}\n"
        f"DOMINANT_CHARGE: {dc}\n"
        f"REASONING_CONFIDENCE: {confidence}"
    )


# ---- numeric standardization stats ------------------------------------------

@dataclass
class NumericStats:
    """Train-only mean/std for the 3 continuous numeric targets."""
    log_vo_mean: float
    log_vo_std: float
    log_K_eq_mean: float
    log_K_eq_std: float
    delta_Ef_mean: float
    delta_Ef_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "log_vo_mean": self.log_vo_mean,
            "log_vo_std": self.log_vo_std,
            "log_K_eq_mean": self.log_K_eq_mean,
            "log_K_eq_std": self.log_K_eq_std,
            "delta_Ef_mean": self.delta_Ef_mean,
            "delta_Ef_std": self.delta_Ef_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NumericStats":
        return cls(**d)

    def standardize(self, log_vo, log_K_eq, delta_Ef) -> tuple[float, float, float]:
        return (
            (float(log_vo) - self.log_vo_mean) / max(self.log_vo_std, 1e-6),
            (float(log_K_eq) - self.log_K_eq_mean) / max(self.log_K_eq_std, 1e-6),
            (float(delta_Ef) - self.delta_Ef_mean) / max(self.delta_Ef_std, 1e-6),
        )

    def unstandardize(self, z: np.ndarray | torch.Tensor, key: str):
        m = getattr(self, f"{key}_mean"); s = getattr(self, f"{key}_std")
        return z * max(s, 1e-6) + m


def compute_numeric_stats(jsonl_path: str | Path) -> NumericStats:
    """Read the TRAIN jsonl and compute per-target mean/std.  Includes the
    log_vo=5.0 floor labels because they are physically meaningful (low
    concentration regime). The histogram is logged separately by the trainer.
    """
    log_vos, log_Ks, dEfs = [], [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            r = json.loads(line)
            log_vos.append(float(r["label"]["log_vo_predicted"]))
            log_Ks.append(float(r["label"]["log_K_eq"]))
            dEfs.append(float(r["label"]["delta_E_f_eV"]))
    log_vos = np.asarray(log_vos, dtype=np.float64)
    log_Ks = np.asarray(log_Ks, dtype=np.float64)
    dEfs = np.asarray(dEfs, dtype=np.float64)
    return NumericStats(
        log_vo_mean=float(log_vos.mean()), log_vo_std=float(log_vos.std() or 1.0),
        log_K_eq_mean=float(log_Ks.mean()), log_K_eq_std=float(log_Ks.std() or 1.0),
        delta_Ef_mean=float(dEfs.mean()), delta_Ef_std=float(dEfs.std() or 1.0),
    )


# ---- the Dataset -------------------------------------------------------------

class Stage1QADataset(Dataset):
    """Causal-LM dataset over the Stage-1 QA pairs.

    Each item yields:
      input_ids        : LongTensor [T]   (prompt+answer, truncated to max_len)
      attention_mask   : LongTensor [T]   (1 for real tokens)
      labels           : LongTensor [T]   (-100 on prompt tokens & padding)
      numeric_targets  : FloatTensor [3]  standardized (log_vo, log_K_eq, dEf)
      charge_target    : LongTensor []    in {0,1,2} (V_O dominant charge)
      floor_mask       : FloatTensor []   1.0 if log_vo == 5.0 (floor pile-up)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        numeric_stats: NumericStats | None = None,
        max_len: int = 2048,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.numeric_stats: NumericStats | None = numeric_stats
        self.records: list[dict] = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        # tokenization cache: only the prompt portion length per sample is
        # variable-but-cheap; we tokenize on the fly in __getitem__.

    # API ---------------------------------------------------------------------

    def set_numeric_stats(self, stats: NumericStats) -> None:
        self.numeric_stats = stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if self.numeric_stats is None:
            raise RuntimeError(
                "Stage1QADataset: numeric_stats not set; call set_numeric_stats "
                "(or pass stats= at construction time)."
            )
        r = self.records[idx]
        prompt = build_prompt(r["input"])
        completion = build_completion(r["label"])

        # We use Qwen2.5's chat template for the prompt only (so the model sees
        # a familiar conversational prefix), then append the completion verbatim.
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # tokenizer without chat template (very rare for Qwen2.5)
            prompt_text = prompt + "\n"

        full_text = prompt_text + completion + self.tokenizer.eos_token

        # Tokenize prompt and full text separately to find the boundary.
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, truncation=True,
            max_length=self.max_len,
        )["input_ids"]
        full = self.tokenizer(
            full_text, add_special_tokens=False, truncation=True,
            max_length=self.max_len,
        )
        input_ids = full["input_ids"]
        attn = full["attention_mask"]

        # Build labels: copy input_ids, then mask the prompt prefix with -100.
        labels = list(input_ids)
        n_prompt = min(len(prompt_ids), len(labels))
        for i in range(n_prompt):
            labels[i] = -100

        # Sanity: if the completion got entirely truncated, at least supervise
        # the last token so loss isn't all -100 (degenerate -> NaN in HF).
        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        log_vo = float(r["label"]["log_vo_predicted"])
        log_K = float(r["label"]["log_K_eq"])
        dEf = float(r["label"]["delta_E_f_eV"])
        z_vo, z_K, z_E = self.numeric_stats.standardize(log_vo, log_K, dEf)

        dc_raw = int(r["label"]["dominant_charge_state"])
        # charge labels are typically 0/+1/+2 ; map to class index in {0,1,2}.
        charge_idx = max(0, min(2, dc_raw))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "numeric_targets": torch.tensor([z_vo, z_K, z_E], dtype=torch.float32),
            "charge_target": torch.tensor(charge_idx, dtype=torch.long),
            "floor_mask": torch.tensor(1.0 if abs(log_vo - 5.0) < 1e-6 else 0.0,
                                       dtype=torch.float32),
        }


# ---- collate ----------------------------------------------------------------

def make_collate_fn(pad_token_id: int):
    """Right-pad to longest in batch.  labels padded with -100, attn with 0."""

    def _collate(batch: Iterable[dict]) -> dict:
        batch = list(batch)
        T = max(b["input_ids"].size(0) for b in batch)

        def pad(t: torch.Tensor, val: int, dtype=None) -> torch.Tensor:
            if dtype is None:
                dtype = t.dtype
            out = torch.full((T,), val, dtype=dtype)
            out[: t.size(0)] = t
            return out

        ids = torch.stack([pad(b["input_ids"], pad_token_id) for b in batch], 0)
        attn = torch.stack([pad(b["attention_mask"], 0) for b in batch], 0)
        labels = torch.stack([pad(b["labels"], -100) for b in batch], 0)
        numerics = torch.stack([b["numeric_targets"] for b in batch], 0)
        charge = torch.stack([b["charge_target"] for b in batch], 0)
        floor = torch.stack([b["floor_mask"] for b in batch], 0)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "numeric_targets": numerics,
            "charge_target": charge,
            "floor_mask": floor,
        }

    return _collate
