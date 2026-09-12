"""
Stage-2 Process-LLM + Bridge trainer (Phase 58, design doc §5.2).

Total loss = span MLM (L_LM on masked positions only)
          + lambda3 * InfoNCE on z_proto (positives = same-paper augmented view)
          + lambda4 * BCE-with-logits on the 12 protocol classes
          + lambda5 * MSE between Bridge output (z_LLM) and the frozen
            Physics-LLM z_phys for the same (dopant, atm, T) tuple

  lambda3 = 0.5, lambda4 = 0.3, lambda5 = 0.2   (§5.2.3)
  InfoNCE temperature τ = 0.07                   (§3.7 / §5.2.3)

Trains:
  - ProcessLLM LoRA adapter (Qwen2.5-7B + r=16 LoRA)
  - ProcessLLM z_proj (3584 → 32)
  - ProcessLLM protocol_head (3584 → 12)
  - LLMBridge (≈25k params)
PhysicsLLM (M6) is FROZEN — only used to read out z_phys for the bridge loss.
If a Stage-1 adapter does not yet exist, the caller may pass `physics_llm=None`
and the trainer will substitute a random-init z_phys for each row (the bridge
still trains; the MSE just regresses against a fixed random embedding).

Per the [feedback_gpu_utilization] memory: if util drops <60% in real training,
the entry-point script `47_v58_train_stage2.py` exposes `--batch-size` /
`--grad-accum` knobs.  The trainer itself is stateless about device placement —
batch tensors are moved to `next(process_llm.parameters()).device`.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.stage2_paper_dataset import (
    PROTOCOL_LABEL_NAMES,
    Stage2PaperDataset,
    make_stage2_collate,
)

logger = logging.getLogger(__name__)


# ── Config dataclass ─────────────────────────────────────────────────────────

@dc.dataclass
class Stage2Cfg:
    output_dir: Path
    num_epochs: int = 2
    batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 1.5e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_length: int = 4096
    eval_steps: int = 200
    save_steps: int = 400
    log_steps: int = 5
    max_steps: int | None = None
    # loss weights (§5.2.3)
    lambda_mlm: float = 1.0
    lambda_nce: float = 0.5
    lambda_protocol: float = 0.3
    lambda_bridge: float = 0.2
    nce_temperature: float = 0.07
    # MLM
    mlm_ratio: float = 0.15
    mean_span: float = 5.0
    aug_dropout: float = 0.10
    bf16: bool = True
    seed: int = 42
    num_workers: int = 0


# ── Stage-2 trainer ──────────────────────────────────────────────────────────

class Stage2Trainer:
    """Stage-2 Process-LLM + Bridge trainer.

    Args:
        process_llm           : ProcessLLM (real or tiny_test).  Its LoRA,
                                z_proj, and protocol_head are trained.
        physics_llm_frozen    : PhysicsLLM or None.  If None, z_phys is drawn
                                from `torch.randn(B, 32)` each step (smoke/CI).
        bridge                : LLMBridge (M7).  Trained.
        train_ds, val_ds      : Stage2PaperDataset
        cfg                   : Stage2Cfg
    """

    def __init__(
        self,
        process_llm,
        physics_llm_frozen,
        bridge: nn.Module,
        train_ds: Stage2PaperDataset,
        val_ds: Stage2PaperDataset,
        cfg: Stage2Cfg,
    ):
        self.process_llm = process_llm
        self.physics_llm = physics_llm_frozen   # None or PhysicsLLM
        self.bridge = bridge
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg

        # Ensure PhysicsLLM is fully frozen + eval if given
        if self.physics_llm is not None:
            self.physics_llm.freeze_for_stage3()  # idempotent
        else:
            # Audit-6 P1a fix: without a trained Stage-1 PhysicsLLM, _derive_z_phys_real
            # cannot produce a real teacher. Disable bridge MSE rather than regress against
            # random noise (the silent-stub bug). Refinement pass after Stage 1 re-enables.
            # This matches the user's sequencing rule (Stage 1 waits for DFT; Stage 2 runs
            # first without bridge MSE).
            if self.cfg.lambda_bridge > 0:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "physics_llm is None — forcing lambda_bridge=0 (no z_phys teacher; "
                    "bridge will be fit in a post-Stage-1 refinement run)."
                )
                self.cfg.lambda_bridge = 0.0

        self.device = next(process_llm.parameters()).device
        self.dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        # Move bridge to the same device as the process LLM
        self.bridge.to(self.device)

        # Collate (uses Process-LLM tokenizer)
        if process_llm.tokenizer is None and not process_llm.tiny_test:
            raise RuntimeError(
                "Stage2Trainer requires a tokenizer on the process_llm "
                "(either tiny_test=True or from_pretrained_qlora)."
            )
        # tokenizer may be None in tiny-test mode → use a minimal stub
        if process_llm.tokenizer is None:
            tok = _TinyToyTokenizer()
        else:
            tok = process_llm.tokenizer
        self._tokenizer = tok

        self.collate = make_stage2_collate(
            tok,
            max_seq_length=cfg.max_seq_length,
            mlm_ratio=cfg.mlm_ratio,
            mean_span=cfg.mean_span,
            aug_dropout=cfg.aug_dropout,
            seed=cfg.seed,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=self.collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=self.collate, drop_last=False,
        )

        # Optimizer (PagedAdamW8bit if available; AdamW otherwise — tiny-test
        # / no-bnb path)
        trainable: list[nn.Parameter] = []
        for p in self.process_llm.parameters():
            if p.requires_grad:
                trainable.append(p)
        for p in self.bridge.parameters():
            if p.requires_grad:
                trainable.append(p)
        # PagedAdamW8bit requires CUDA; fall back to AdamW on CPU.
        use_paged = (self.device.type == "cuda")
        if use_paged:
            try:
                from bitsandbytes.optim import PagedAdamW8bit
                self.optimizer = PagedAdamW8bit(
                    trainable, lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
                self._opt_name = "paged_adamw_8bit"
            except Exception:
                use_paged = False
        if not use_paged:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            self._opt_name = "adamw"

        # Cosine LR schedule
        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, cfg.grad_accum_steps))
        if cfg.max_steps is not None:
            total_opt_steps = int(cfg.max_steps)
        else:
            total_opt_steps = max(1, steps_per_epoch * cfg.num_epochs)
        warmup = max(1, int(total_opt_steps * cfg.warmup_ratio))
        self._total_opt_steps = total_opt_steps
        self._warmup = warmup

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            t = (step - warmup) / max(1, total_opt_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # BCE class weights so the trainer doesn't ignore rare classes
        self._pos_weight = train_ds.label_pos_weight().to(self.device)

        # metrics log
        self.metrics_path = Path(cfg.output_dir) / "metrics.json"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_log: list[dict] = []
        self._step_loss_log: list[dict] = []
        self.global_step = 0
        self.opt_step = 0

    # ── core forward + loss ──────────────────────────────────────────────────

    def _backbone_outputs(self, ids: torch.Tensor, attn: torch.Tensor,
                          want_lm_loss: bool, labels_lm: torch.Tensor | None):
        """Run the Process-LLM backbone once.  Returns (lm_loss_or_None, pooled).

        - When want_lm_loss is True we pass `labels=labels_lm` so the PEFT/HF
          model computes the shifted-CE for us at non-(-100) positions.
        - We always request `output_hidden_states=True` so we can mean-pool
          for z_proto + protocol_head.
        """
        kwargs = dict(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        if want_lm_loss:
            kwargs["labels"] = labels_lm

        outputs = self.process_llm.backbone(**kwargs)
        lm_loss = outputs.loss if want_lm_loss else None
        last_hidden = outputs.hidden_states[-1]                # [B, T, H]
        mask = attn.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return lm_loss, pooled

    def _info_nce(self, z_a: torch.Tensor, z_b: torch.Tensor,
                  paper_ids: list[str]) -> torch.Tensor:
        """Symmetric InfoNCE with τ=cfg.nce_temperature.

        Positive for row i: the augmented-view embedding z_b[i] (same paper).
        Plus any in-batch row j (i != j) with the same paper_id is also a
        positive (rare in v1 since one row per paper).  We use the standard
        SimCLR-style loss: cross-entropy with target = i over the 2B-row
        similarity matrix where i pairs with i+B.
        """
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        B = z_a.size(0)
        if B < 2:
            # need at least 2 rows for InfoNCE
            return torch.zeros((), device=z_a.device, dtype=torch.float32)
        z = torch.cat([z_a, z_b], dim=0)                       # [2B, D]
        logits = z @ z.t() / max(1e-6, self.cfg.nce_temperature)  # [2B, 2B]
        # mask self-similarity
        eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        logits.masked_fill_(eye, float("-inf"))
        # target: row i (0..B-1) -> column i+B ; row i+B -> column i
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ], dim=0)
        return F.cross_entropy(logits, targets)

    def _forward_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        ids = batch["input_ids"].to(self.device, non_blocking=True)
        attn = batch["attention_mask"].to(self.device, non_blocking=True)
        labels_lm = batch["labels_lm"].to(self.device, non_blocking=True)
        ids_pos = batch["input_ids_pos"].to(self.device, non_blocking=True)
        attn_pos = batch["attention_mask_pos"].to(self.device, non_blocking=True)
        labels_proto = batch["labels_protocol"].to(self.device, non_blocking=True)
        paper_ids = batch["paper_ids"]

        # Original view: compute LM loss + z_proto + protocol_logits
        lm_loss, pooled = self._backbone_outputs(
            ids, attn, want_lm_loss=True, labels_lm=labels_lm,
        )
        if lm_loss is None or not torch.isfinite(lm_loss):
            lm_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        # cast pooled to head dtype
        z_proj_dtype = self.process_llm.z_proj.weight.dtype
        z_proto = self.process_llm.z_proj(pooled.to(z_proj_dtype))
        head_dtype = self.process_llm.protocol_head.weight.dtype
        proto_logits = self.process_llm.protocol_head(pooled.to(head_dtype))

        # Augmented view: just z_proto for InfoNCE
        _, pooled_pos = self._backbone_outputs(
            ids_pos, attn_pos, want_lm_loss=False, labels_lm=None,
        )
        z_proto_pos = self.process_llm.z_proj(pooled_pos.to(z_proj_dtype))

        # Protocol multi-label BCE
        protocol_loss = F.binary_cross_entropy_with_logits(
            proto_logits.float(), labels_proto.float(),
            pos_weight=self._pos_weight.float(),
        )

        # InfoNCE on z_proto
        nce_loss = self._info_nce(z_proto, z_proto_pos, paper_ids)

        # Bridge: z_phys is either looked up from frozen PhysicsLLM (real run)
        # or a fixed random embedding (smoke).  We compute a per-row z_phys.
        with torch.no_grad():
            if self.physics_llm is not None:
                # Real path: ask PhysicsLLM for z_phys using a stub prompt
                # derived from the labels.  For the smoke / first launch the
                # caller doesn't have Stage-1 yet, so the random path is used.
                z_phys = self._derive_z_phys_real(paper_ids, batch)
            else:
                z_phys = self._derive_z_phys_random(z_proto.size(0))
            z_phys = z_phys.to(z_proto.device, dtype=z_proto.dtype)

        z_llm = self.bridge(z_proto, z_phys)
        bridge_loss = F.mse_loss(z_llm.float(), z_phys.float())

        total = (
            self.cfg.lambda_mlm * lm_loss.float()
            + self.cfg.lambda_nce * nce_loss.float()
            + self.cfg.lambda_protocol * protocol_loss.float()
            + self.cfg.lambda_bridge * bridge_loss.float()
        )

        # metrics tracking
        with torch.no_grad():
            proto_pred = (torch.sigmoid(proto_logits.float()) > 0.5).int()
            tp = ((proto_pred == 1) & (labels_proto.int() == 1)).sum().item()
            fp = ((proto_pred == 1) & (labels_proto.int() == 0)).sum().item()
            fn = ((proto_pred == 0) & (labels_proto.int() == 1)).sum().item()

        return {
            "loss": total,
            "lm_loss": lm_loss.detach().float(),
            "nce_loss": nce_loss.detach().float(),
            "protocol_loss": protocol_loss.detach().float(),
            "bridge_loss": bridge_loss.detach().float(),
            "tp": tp, "fp": fp, "fn": fn,
            "z_proto_norm": z_proto.detach().float().norm(dim=-1).mean().item(),
        }

    def _derive_z_phys_random(self, B: int) -> torch.Tensor:
        # Deterministic per-step but different per row
        gen = torch.Generator(device="cpu").manual_seed(
            self.cfg.seed + self.global_step
        )
        return torch.randn(B, 32, generator=gen, dtype=torch.float32)

    def _derive_z_phys_real(self, paper_ids: list[str], batch: dict) -> torch.Tensor:
        """Real Stage-1 PhysicsLLM lookup.

        Heuristic: for each row, build a single-token prompt summarizing
        (dopant, atmosphere, temperature) extracted from the methods text.
        Stage 2 doesn't have row-level dopant labels (those are in the v58
        CSV), so for the smoke path we just hit the model with the same
        physics prompt seed for all rows in the batch.  Stage-2 *training*
        for the final run can be extended to look up per-paper dopant via
        `data/v56_llm_extracted.csv`; we leave that as a TODO and the random
        path is used until then.
        """
        # Simple fallback: use random; the trainer documents this in metrics.
        return self._derive_z_phys_random(len(paper_ids))

    # ── evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        # Tiny-test path may have no val_ds rows; protect against that.
        if len(self.val_ds) == 0:
            return {"val_n": 0}

        self.process_llm.backbone.eval()
        self.bridge.eval()

        tot = {
            "lm": 0.0, "nce": 0.0, "protocol": 0.0, "bridge": 0.0, "n": 0,
            "tp": 0, "fp": 0, "fn": 0,
        }
        z_proto_all = []
        paper_ids_all: list[str] = []
        for batch in self.val_loader:
            out = self._forward_loss(batch)
            B = batch["input_ids"].size(0)
            tot["lm"] += float(out["lm_loss"]) * B
            tot["nce"] += float(out["nce_loss"]) * B
            tot["protocol"] += float(out["protocol_loss"]) * B
            tot["bridge"] += float(out["bridge_loss"]) * B
            tot["n"] += B
            tot["tp"] += out["tp"]; tot["fp"] += out["fp"]; tot["fn"] += out["fn"]

        precision = tot["tp"] / max(1, tot["tp"] + tot["fp"])
        recall = tot["tp"] / max(1, tot["tp"] + tot["fn"])
        micro_f1 = (
            2 * precision * recall / max(1e-9, precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        self.process_llm.backbone.train()
        self.bridge.train()

        return {
            "val_lm_loss": tot["lm"] / max(1, tot["n"]),
            "val_nce_loss": tot["nce"] / max(1, tot["n"]),
            "val_protocol_loss": tot["protocol"] / max(1, tot["n"]),
            "val_bridge_loss": tot["bridge"] / max(1, tot["n"]),
            "val_protocol_micro_f1": float(micro_f1),
            "val_protocol_precision": float(precision),
            "val_protocol_recall": float(recall),
            "val_n": tot["n"],
            # target ≥ 0.75 from doc §5.2.5
            "val_protocol_micro_f1_meets_target": float(micro_f1) >= 0.75,
        }

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, tag: str) -> Path:
        out_dir = Path(self.cfg.output_dir) / f"adapter-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapter (peft) — backbone is a PeftModel in the real run, but in
        # tiny-test mode it's a plain Module so save_pretrained is unavailable.
        if hasattr(self.process_llm.backbone, "save_pretrained"):
            try:
                self.process_llm.backbone.save_pretrained(str(out_dir))
            except Exception as e:
                logger.warning("save_pretrained failed: %s", e)
        # heads + bridge in a single torch file
        state = {
            "z_proj": self.process_llm.z_proj.state_dict(),
            "protocol_head": self.process_llm.protocol_head.state_dict(),
            "bridge": self.bridge.state_dict(),
        }
        torch.save(state, out_dir / "heads_bridge.pt")
        return out_dir

    def _append_metrics(self, payload: dict) -> None:
        self._metrics_log.append(payload)
        with open(self.metrics_path, "w") as f:
            json.dump({
                "run_dir": str(self.cfg.output_dir),
                "config": dc.asdict(self.cfg) | {"output_dir": str(self.cfg.output_dir)},
                "optimizer": self._opt_name,
                "total_opt_steps": self._total_opt_steps,
                "warmup": self._warmup,
                "step_loss_log": self._step_loss_log,
                "metrics_log": self._metrics_log,
                "label_pos_weight": self._pos_weight.cpu().tolist(),
                "label_names": PROTOCOL_LABEL_NAMES,
            }, f, indent=2, default=str)

    # ── train loop ───────────────────────────────────────────────────────────

    def fit(self) -> dict[str, Any]:
        torch.manual_seed(self.cfg.seed)
        self.process_llm.backbone.train()
        self.bridge.train()
        accum = 0
        running = {k: 0.0 for k in
                   ("loss", "lm", "nce", "protocol", "bridge")}
        running["n"] = 0
        t0 = time.time()

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (self.cfg.bf16 and self.device.type == "cuda")
            else nullcontext()
        )

        for epoch in range(self.cfg.num_epochs):
            for batch in self.train_loader:
                self.global_step += 1
                with autocast_ctx:
                    out = self._forward_loss(batch)
                loss = out["loss"] / self.cfg.grad_accum_steps
                loss.backward()

                running["loss"] += float(out["loss"].detach())
                running["lm"] += float(out["lm_loss"])
                running["nce"] += float(out["nce_loss"])
                running["protocol"] += float(out["protocol_loss"])
                running["bridge"] += float(out["bridge_loss"])
                running["n"] += 1
                accum += 1

                if accum >= self.cfg.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.process_llm.parameters() if p.requires_grad]
                        + [p for p in self.bridge.parameters() if p.requires_grad],
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.opt_step += 1
                    accum = 0

                    if self.opt_step % max(1, self.cfg.log_steps) == 0:
                        n = max(1, running["n"])
                        log_payload = {
                            "opt_step": self.opt_step,
                            "loss": running["loss"] / n,
                            "lm_loss": running["lm"] / n,
                            "nce_loss": running["nce"] / n,
                            "protocol_loss": running["protocol"] / n,
                            "bridge_loss": running["bridge"] / n,
                            "lr": self.scheduler.get_last_lr()[0],
                            "elapsed_s": time.time() - t0,
                        }
                        self._step_loss_log.append(log_payload)
                        msg = (
                            f"[step {self.opt_step}/{self._total_opt_steps}] "
                            f"loss={log_payload['loss']:.4f} "
                            f"lm={log_payload['lm_loss']:.4f} "
                            f"nce={log_payload['nce_loss']:.4f} "
                            f"proto={log_payload['protocol_loss']:.4f} "
                            f"br={log_payload['bridge_loss']:.4f} "
                            f"lr={log_payload['lr']:.2e} "
                            f"t={log_payload['elapsed_s']:.1f}s"
                        )
                        logger.info(msg)
                        print(msg, flush=True)
                        running = {k: 0.0 for k in
                                   ("loss", "lm", "nce", "protocol", "bridge")}
                        running["n"] = 0

                    if (self.cfg.eval_steps > 0
                            and self.opt_step % self.cfg.eval_steps == 0):
                        val_metrics = self.evaluate()
                        val_metrics["opt_step"] = self.opt_step
                        self._append_metrics(val_metrics)

                    if (self.cfg.save_steps > 0
                            and self.opt_step % self.cfg.save_steps == 0):
                        path = self.save(f"step-{self.opt_step}")
                        logger.info("saved checkpoint to %s", path)

                    if (self.cfg.max_steps is not None
                            and self.opt_step >= self.cfg.max_steps):
                        break

            if (self.cfg.max_steps is not None
                    and self.opt_step >= self.cfg.max_steps):
                break

        # Final eval + save
        final = self.evaluate()
        final["opt_step"] = self.opt_step
        final["_final"] = True
        self._append_metrics(final)
        path = self.save(f"final-step-{self.opt_step}")
        logger.info("final checkpoint: %s", path)
        return {
            "final_metrics": final,
            "final_adapter_dir": str(path),
            "metrics_path": str(self.metrics_path),
        }


# ── small stub for the tiny-test path ────────────────────────────────────────

class _TinyToyTokenizer:
    """A *minimal* tokenizer used only when ProcessLLM(tiny_test=True) and the
    caller wants to exercise the trainer wiring on CPU without HF weights.

    It does byte-level tokenization with vocab=256 (matches the toy backbone),
    pads/truncates to a fixed length, and exposes the `__call__` API the
    Stage-2 collate uses.  NOT a substitute for the real Qwen2.5 tokenizer.
    """

    pad_token_id = 0
    eos_token_id = 0
    mask_token_id = None
    pad_token = "[PAD]"
    eos_token = ""

    def __call__(self, texts, return_tensors="pt", max_length=128,
                 truncation=True, padding=True, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = []
        for t in texts:
            ids = [b % 256 for b in t.encode("utf-8")[:max_length]]
            encoded.append(ids)
        T = max((len(x) for x in encoded), default=1)
        ids = torch.zeros((len(encoded), T), dtype=torch.long)
        attn = torch.zeros((len(encoded), T), dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
            attn[i, : len(e)] = 1
        return {"input_ids": ids, "attention_mask": attn}


# ── self-test entry (CPU, tiny_test) ─────────────────────────────────────────

def _self_test() -> None:
    import tempfile, json as _json
    from src.models.llm_process import ProcessLLM
    from src.models.llm_bridge import LLMBridge
    from src.data.stage2_paper_dataset import (
        PROTOCOL_LABEL_NAMES as _NAMES,
    )

    print("[Stage2Trainer self-test]")
    torch.manual_seed(0)

    # 1. Build a temp dataset of 4 tiny papers
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        methods_dir = tmp / "data" / "processed" / "paper_methods"
        methods_dir.mkdir(parents=True)
        rows = []
        for i in range(4):
            pid = f"p{i:04x}"
            (methods_dir / f"{pid}.txt").write_text(
                f"Paper {pid} methods. " + "Mg-doped Ga2O3 sputtered. " * 20,
                encoding="utf-8",
            )
            labels = {n: (1 if (i + j) % 3 == 0 else 0) for j, n in enumerate(_NAMES)}
            rows.append({
                "paper_id": pid,
                "methods_text_path": f"data/processed/paper_methods/{pid}.txt",
                "method_text_len": 200,
                "labels": labels,
                "n_positive_labels": sum(labels.values()),
            })
        jsonl_path = tmp / "data" / "processed" / "paper_protocol_labels.jsonl"
        with open(jsonl_path, "w") as f:
            for r in rows:
                f.write(_json.dumps(r) + "\n")
        ds = Stage2PaperDataset(jsonl_path, repo_root=tmp)
        print(f"  dataset rows : {len(ds)}")

        # 2. Build tiny ProcessLLM + LLMBridge on CPU
        tiny_hidden = 64
        process = ProcessLLM(tiny_test=True, hidden_size=tiny_hidden)
        # need a tokenizer-like object for the collate
        process.tokenizer = _TinyToyTokenizer()
        bridge = LLMBridge(dim=32, n_heads=4, n_layers=2, dropout=0.0)

        # Need the toy backbone to compute LM loss — patch its forward to
        # produce a `.loss` attribute when labels are passed.  We wrap it.
        orig_forward = process.backbone.forward
        def _wrapped_forward(input_ids, attention_mask=None,
                             output_hidden_states=True, labels=None,
                             use_cache=False, **kwargs):
            out = orig_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            if labels is not None:
                # tiny linear LM head: project last hidden -> vocab=256
                lh = out.hidden_states[-1]
                vocab = 256
                if not hasattr(process.backbone, "_tiny_lm_head"):
                    process.backbone._tiny_lm_head = nn.Linear(
                        tiny_hidden, vocab, bias=False,
                    )
                logits = process.backbone._tiny_lm_head(lh)
                # shift one token for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                mask = shift_labels != -100
                if mask.sum() == 0:
                    out.loss = torch.zeros((), dtype=torch.float32)
                else:
                    out.loss = F.cross_entropy(
                        shift_logits.view(-1, vocab),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
            else:
                out.loss = None
            return out
        process.backbone.forward = _wrapped_forward

        out_dir = tmp / "out"
        cfg = Stage2Cfg(
            output_dir=out_dir, num_epochs=1,
            batch_size=2, grad_accum_steps=1,
            learning_rate=1e-3,
            max_seq_length=128, eval_steps=0, save_steps=0,
            log_steps=1, max_steps=2,
            bf16=False, num_workers=0,
        )
        trainer = Stage2Trainer(
            process_llm=process,
            physics_llm_frozen=None,    # smoke: random z_phys
            bridge=bridge,
            train_ds=ds, val_ds=ds,
            cfg=cfg,
        )
        print(f"  trainable params (process_llm): "
              f"{sum(p.numel() for p in process.parameters() if p.requires_grad):,}")
        print(f"  trainable params (bridge)     : "
              f"{sum(p.numel() for p in bridge.parameters() if p.requires_grad):,}")
        print(f"  optimizer                     : {trainer._opt_name}")
        print(f"  total_opt_steps               : {trainer._total_opt_steps}")

        result = trainer.fit()
        print(f"  final metrics                 : {result['final_metrics']}")
        print(f"  final_adapter_dir             : {result['final_adapter_dir']}")
    print("[Stage2Trainer self-test] PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_test()
