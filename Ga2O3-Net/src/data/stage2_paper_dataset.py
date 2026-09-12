"""
Phase-58 Stage-2 paper dataset.

Wraps `data/processed/paper_protocol_labels.jsonl` (output of
`scripts/46_v58_assign_protocol_labels.py`) for the Stage-2 Process-LLM
trainer (§5.2 of the design doc).

Each item exposes:
  - methods_text       : str (the paper's extracted Methods/Experimental text)
  - labels             : FloatTensor [12] (binary protocol multi-label)
  - paper_id           : str
  - positive_paper_ids : list[str] — same-paper InfoNCE positives.  Since
    each paper contributes ONE row in this v1 implementation (one extracted
    Methods text per paper), we synthesize positives by sampling a second
    augmented view of the SAME paper (drop-out span mask of 10% — different
    seed than the MLM-target mask).  This gives every paper a self-positive
    pair, which is sufficient for in-batch InfoNCE (§5.2.3 — "positive = same
    paper, hard negative = cross-paper same dopant; in-batch negatives").

The collate_fn:
  - Tokenizes BOTH the original methods text AND an "augmented view" via
    the Qwen2.5-7B tokenizer (max_len 4096).  The augmented view differs
    from the MLM-target text only in random subword dropout (~10%).
  - For span MLM (§5.2.3): mask ~15% of NON-pad input_ids in the ORIGINAL
    sequence with a span-length Bernoulli (geometric, p=0.2 → mean span ~5
    tokens); replaced positions are set to [MASK]/eos depending on tokenizer.
    Labels are -100 EXCEPT at masked positions (loss only on masks).
  - Returns ready-to-feed batch dict for the trainer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


PROTOCOL_LABEL_NAMES = [
    "target_purity_4N+",
    "Ar_O2_ratio_oxidizing",
    "RF_power_high",
    "substrate_sapphire",
    "anneal_atm_O2",
    "anneal_atm_N2",
    "contact_metal_TiAu",
    "contact_metal_other",
    "illum_254nm",
    "illum_365nm",
    "bias_self_powered",
    "bias_>=5V",
]


@dataclass
class _Sample:
    paper_id: str
    methods_text_path: str
    methods_text: str
    labels: torch.Tensor          # [12]
    method_text_len: int


class Stage2PaperDataset(Dataset):
    """Dataset of (paper Methods text, 12 protocol labels).

    Args:
        labels_jsonl : path to `paper_protocol_labels.jsonl` (script 46).
        repo_root    : path that the `methods_text_path` fields are relative
                       to (default: project root inferred from this file).
        max_chars    : safety truncation on raw text BEFORE tokenization
                       (the tokenizer further truncates to max_seq_length).
                       Default 80_000 chars (Qwen tokenizer roughly 4 chars/
                       token → ~20k tokens; the 4096-token max in the tokenizer
                       handles the rest).
    """

    def __init__(
        self,
        labels_jsonl: str | Path,
        repo_root: str | Path | None = None,
        max_chars: int = 80_000,
        skip_zero_positive: bool = False,
    ):
        self.path = Path(labels_jsonl)
        if repo_root is None:
            # default: 3 parents up from this file (src/data/file.py → repo)
            repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = Path(repo_root)
        self.max_chars = int(max_chars)
        self.samples: list[_Sample] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if skip_zero_positive and row["n_positive_labels"] == 0:
                    continue
                txt_path = self.repo_root / row["methods_text_path"]
                try:
                    text = txt_path.read_text(encoding="utf-8", errors="ignore")
                except FileNotFoundError:
                    continue
                if len(text) > self.max_chars:
                    text = text[: self.max_chars]
                labels = [int(row["labels"][n]) for n in PROTOCOL_LABEL_NAMES]
                self.samples.append(_Sample(
                    paper_id=row["paper_id"],
                    methods_text_path=row["methods_text_path"],
                    methods_text=text,
                    labels=torch.tensor(labels, dtype=torch.float32),
                    method_text_len=int(row["method_text_len"]),
                ))

        # paper_id -> index for quick lookup of positives
        self._paper_to_idx = {s.paper_id: i for i, s in enumerate(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        # Positive partner for InfoNCE: in this v1 we self-positive (same paper,
        # a 10%-dropout augmented view created inside collate_fn at tokenize
        # time).  The trainer can also do in-batch positives when two rows
        # share the same paper_id; we expose paper_id so it can do that.
        return {
            "paper_id": s.paper_id,
            "methods_text": s.methods_text,
            "labels": s.labels,
            "positive_paper_ids": [s.paper_id],  # self-positive (augmented view)
            "method_text_len": s.method_text_len,
        }

    # Statistics helpers used by the trainer + report ─────────────────────────

    def n_papers(self) -> int:
        return len({s.paper_id for s in self.samples})

    def label_positive_counts(self) -> dict[str, int]:
        counts = {n: 0 for n in PROTOCOL_LABEL_NAMES}
        for s in self.samples:
            for i, n in enumerate(PROTOCOL_LABEL_NAMES):
                counts[n] += int(s.labels[i].item())
        return counts

    def label_pos_weight(self) -> torch.Tensor:
        """BCE-with-logits pos_weight = (n_neg / n_pos) per class, clipped to
        [1, 50] so all-zero or near-all-zero classes don't dominate."""
        N = len(self.samples)
        out = []
        for i in range(len(PROTOCOL_LABEL_NAMES)):
            n_pos = int(sum(int(s.labels[i].item()) for s in self.samples))
            n_neg = N - n_pos
            w = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
            out.append(float(min(50.0, max(1.0, w))))
        return torch.tensor(out, dtype=torch.float32)

    def n_positive_pairs(self) -> int:
        """How many same-paper InfoNCE positive pairs exist.

        v1: each paper has 1 row, so we always synthesize 1 augmented partner
        per paper at collate time.  Total available unique positive pairs is
        therefore equal to the dataset size (each row pairs with its augmented
        view).  Reported for the smoke-deliverable statistics.
        """
        return len(self.samples)


# ── Collate / tokenization for Stage-2 ───────────────────────────────────────

def _geometric_span_lengths(n_tokens: int, mask_ratio: float, mean_span: float,
                            rng: random.Random) -> list[tuple[int, int]]:
    """Return [(start, length), ...] spans covering ~mask_ratio of tokens,
    each span length ~ Geometric(p) with mean ~ mean_span.
    p = 1 / mean_span, length = 1 + Geometric(p) clipped to [1, 10].
    """
    target = max(1, int(round(mask_ratio * n_tokens)))
    p = 1.0 / max(1.5, mean_span)
    spans: list[tuple[int, int]] = []
    covered = 0
    tries = 0
    used = np.zeros(n_tokens, dtype=bool)
    while covered < target and tries < 4 * target:
        tries += 1
        length = 1
        # geometric-like: keep extending with prob (1-p)
        while length < 10 and rng.random() > p:
            length += 1
        start = rng.randint(0, max(0, n_tokens - length))
        end = start + length
        if used[start:end].any():
            continue
        used[start:end] = True
        spans.append((start, length))
        covered += length
    return spans


def make_stage2_collate(tokenizer, max_seq_length: int = 4096,
                        mlm_ratio: float = 0.15, mean_span: float = 5.0,
                        aug_dropout: float = 0.10, seed: int = 0):
    """Build the collate_fn closed over the tokenizer + hyperparameters."""

    # Choose a mask token: Qwen2.5 doesn't have a real [MASK]; use eos_token_id
    # as a stand-in.  The causal-LM forward only sees this token at masked
    # positions and is supervised by the ORIGINAL token via labels.  This is
    # the standard "span-mask + LM-loss-only-on-masked-position" recipe.
    mask_token_id = (
        tokenizer.mask_token_id if getattr(tokenizer, "mask_token_id", None) is not None
        else tokenizer.eos_token_id
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    rng = random.Random(seed)

    def _aug_view(text: str) -> str:
        """Cheap subword-level dropout in CHAR space (drop ~aug_dropout of
        characters in random short runs).  Tokenizes differently from the
        original → a useful contrastive positive."""
        if aug_dropout <= 0.0 or not text:
            return text
        out_chars: list[str] = []
        i = 0
        while i < len(text):
            if rng.random() < aug_dropout / 4:
                # drop a 1-4 char run
                skip = rng.randint(1, 4)
                i += skip
                continue
            out_chars.append(text[i])
            i += 1
        return "".join(out_chars)

    def _collate(batch: Iterable[dict]) -> dict:
        batch = list(batch)
        texts = [b["methods_text"] for b in batch]
        aug_texts = [_aug_view(t) for t in texts]
        labels_protocol = torch.stack([b["labels"] for b in batch], 0)  # [B,12]
        paper_ids = [b["paper_id"] for b in batch]

        enc = tokenizer(
            texts, return_tensors="pt", max_length=max_seq_length,
            truncation=True, padding=True,
        )
        enc_aug = tokenizer(
            aug_texts, return_tensors="pt", max_length=max_seq_length,
            truncation=True, padding=True,
        )

        input_ids = enc["input_ids"].clone()                # [B, T]
        attention_mask = enc["attention_mask"].clone()      # [B, T]
        labels_lm = torch.full_like(input_ids, fill_value=-100)

        # Span-MLM masking on each row (over the real tokens only)
        for b in range(input_ids.size(0)):
            real_len = int(attention_mask[b].sum().item())
            if real_len < 10:
                continue
            spans = _geometric_span_lengths(
                real_len, mlm_ratio, mean_span, rng,
            )
            for start, length in spans:
                end = min(real_len, start + length)
                labels_lm[b, start:end] = input_ids[b, start:end].clone()
                # replace input with mask token id (loss is computed against
                # the original via labels_lm)
                input_ids[b, start:end] = mask_token_id

        return {
            # MLM-side
            "input_ids": input_ids,                # [B, T] (with masks)
            "attention_mask": attention_mask,      # [B, T]
            "labels_lm": labels_lm,                # [B, T] (-100 except masks)
            # InfoNCE-positive view (no MLM mask)
            "input_ids_pos": enc_aug["input_ids"],
            "attention_mask_pos": enc_aug["attention_mask"],
            # protocol multi-label (BCE)
            "labels_protocol": labels_protocol,    # [B, 12]
            # metadata for in-batch positive grouping
            "paper_ids": paper_ids,
        }

    return _collate


def _self_test() -> None:
    """CPU self-test using a stub tokenizer-like object (no HF download)."""
    print("[Stage2PaperDataset self-test]")
    # Build a temp jsonl with one fake row pointing at one fake methods file
    import tempfile, textwrap
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        methods_dir = tmp / "data" / "processed" / "paper_methods"
        methods_dir.mkdir(parents=True)
        (methods_dir / "abc123.txt").write_text(
            "## experimental\n" + "Mg-doped Ga2O3 thin films were deposited by RF magnetron sputtering. " * 50,
            encoding="utf-8",
        )
        jsonl_path = tmp / "data" / "processed" / "paper_protocol_labels.jsonl"
        row = {
            "paper_id": "abc123",
            "methods_text_path": "data/processed/paper_methods/abc123.txt",
            "method_text_len": 100,
            "labels": {n: int(i % 2) for i, n in enumerate(PROTOCOL_LABEL_NAMES)},
            "n_positive_labels": 6,
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(row) + "\n")
        ds = Stage2PaperDataset(jsonl_path, repo_root=tmp)
        assert len(ds) == 1
        item = ds[0]
        assert item["labels"].shape == (12,)
        print(f"  dataset len      : {len(ds)} (OK)")
        print(f"  item labels shape: {item['labels'].shape} (OK)")
        print(f"  positive pairs   : {ds.n_positive_pairs()}")
    print("[Stage2PaperDataset self-test] PASS")


if __name__ == "__main__":
    _self_test()
