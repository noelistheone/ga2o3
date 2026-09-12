"""
Stage-1 Physics-LLM trainer (Phase 58, design doc §5.1).

Wraps `PhysicsLLM` (M6, see src/models/llm_physics.py) for the QLoRA Stage-1
pretraining recipe:

  - loss = L_LM + lambda1 * MSE(numerics) + lambda2 * CE(charge)
    with lambda1=1.0, lambda2=0.3 (§5.1.4).
  - causal LM loss is the standard HF shifted-token CE, computed by the
    PEFT-wrapped backbone (we get logits via output_hidden_states=False on
    the LM path) on the labels mask built by Stage1QADataset (prompt tokens
    masked to -100, only the answer/reasoning region contributes to loss).
  - Numeric prior head + charge head consume the masked mean-pooled last
    hidden state from the SAME forward pass (no extra backbone call).
  - LoRA adapters saved every `save_steps` to checkpoints/.../adapter-step-N.
  - Train/val metrics logged to metrics.json at the run dir (one append per
    eval).
  - At end of training: cross-check on the V55-Ext sputter V_O rows
    (§5.1.6 success criterion).

GPU utilization knob notes (per `feedback_gpu_utilization`):
  - batch_size=1, grad_accum=16 (doc §5.1.5).  Activations are kept tiny by
    Qwen2.5-1.5B + grad-checkpointing; with bf16 + 4-bit base, util on a
    single 3090 sits in the 70-90 % range during the forward+backward step
    and dips during optimizer steps (paged_adamw_8bit). If util stays under
    60 % for many consecutive logging windows, see the entry-point script
    `42_v58_train_stage1.py` for diagnosis flags.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.stage1_qa_dataset import (
    NumericStats,
    Stage1QADataset,
    build_prompt,
    make_collate_fn,
)

logger = logging.getLogger(__name__)


# ---- config helpers ---------------------------------------------------------

@dc.dataclass
class Stage1Cfg:
    output_dir: Path
    num_epochs: int = 3
    batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    eval_steps: int = 200
    save_steps: int = 400
    log_steps: int = 5
    max_steps: int | None = None    # smoke override
    lambda_num: float = 1.0
    lambda_charge: float = 0.3
    bf16: bool = True
    seed: int = 42
    num_workers: int = 0            # batch=1; HF tokenizers fast enough on main
    cross_check_csv: str = "data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv"
    cross_check_n: int = 14         # §5.1.6 ("N=14 sputter V_O rows")


# ---- main trainer -----------------------------------------------------------

class Stage1Trainer:
    """Stage-1 Physics-LLM trainer.

    Args:
        model        : PhysicsLLM (real QLoRA, NOT tiny_test).
        train_ds     : Stage1QADataset (already wired with numeric_stats).
        val_ds       : Stage1QADataset (same stats).
        cfg          : Stage1Cfg (see above).
    """

    def __init__(
        self,
        model,
        train_ds: Stage1QADataset,
        val_ds: Stage1QADataset,
        cfg: Stage1Cfg,
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        self.collate = make_collate_fn(model.tokenizer.pad_token_id)
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=self.collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=self.collate, drop_last=False,
        )

        # paged 8-bit optimizer (§5.1.5)
        from bitsandbytes.optim import PagedAdamW8bit
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = PagedAdamW8bit(
            trainable, lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # cosine schedule
        steps_per_epoch = math.ceil(len(self.train_loader) / cfg.grad_accum_steps)
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

        self.metrics_path = Path(cfg.output_dir) / "metrics.json"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_log: list[dict] = []
        self._step_loss_log: list[dict] = []
        self.global_step = 0
        self.opt_step = 0

    # ---- core forward+loss ---------------------------------------------------

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Single forward producing LM loss + numeric MSE + charge CE.

        We call the PEFT backbone with `labels=` so HF computes the shifted
        CE for us, AND request the last hidden state for the auxiliary heads.
        That is the *same* forward — no second pass.
        """
        ids = batch["input_ids"].to(self.device, non_blocking=True)
        attn = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        numerics = batch["numeric_targets"].to(self.device, non_blocking=True)
        charge_t = batch["charge_target"].to(self.device, non_blocking=True)

        # The PEFT-wrapped model accepts the standard HF args; output_hidden_states
        # exposes the last hidden state for our heads.
        outputs = self.model.backbone(
            input_ids=ids,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        lm_loss = outputs.loss
        if lm_loss is None or torch.isnan(lm_loss):
            # If every label was -100 (shouldn't happen — dataset guarantees
            # at least one supervised token), use a 0 placeholder to avoid
            # propagating NaN through backward.
            lm_loss = torch.zeros((), device=self.device, dtype=self.dtype)

        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        mask = attn.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)  # [B, H]

        # Numeric prior head (§3.6.4): split 6 = 1 + 3 + 1 + 1
        nh = self.model.numeric_head(pooled.to(self.model.numeric_head.net[0].weight.dtype))
        pred_log_vo = nh["log_vo"].squeeze(-1)         # [B]
        pred_charge_logits = nh["charge_logits"]       # [B,3]
        pred_log_K = nh["log_K_eq"].squeeze(-1)        # [B]
        pred_dEf = nh["delta_Ef"].squeeze(-1)          # [B]

        # numerics is stored standardized as [log_vo, log_K_eq, delta_Ef]
        targ_log_vo = numerics[:, 0]
        targ_log_K = numerics[:, 1]
        targ_dEf = numerics[:, 2]

        # NaN-safe MSE: targets and predictions are finite (we standardized
        # on a fully-populated train file), but cast to fp32 for stability.
        mse_log_vo = F.mse_loss(pred_log_vo.float(), targ_log_vo.float())
        mse_log_K = F.mse_loss(pred_log_K.float(), targ_log_K.float())
        mse_dEf = F.mse_loss(pred_dEf.float(), targ_dEf.float())
        num_loss = (mse_log_vo + mse_log_K + mse_dEf) / 3.0

        # charge CE (3-way)
        ce_charge = F.cross_entropy(pred_charge_logits.float(), charge_t)

        total = (
            lm_loss.float()
            + self.cfg.lambda_num * num_loss
            + self.cfg.lambda_charge * ce_charge
        )

        return {
            "loss": total,
            "lm_loss": lm_loss.detach().float(),
            "num_loss": num_loss.detach(),
            "charge_loss": ce_charge.detach(),
            "pred_log_vo_std": pred_log_vo.detach(),
            "targ_log_vo_std": targ_log_vo.detach(),
            "pred_charge": pred_charge_logits.detach().argmax(dim=-1),
            "targ_charge": charge_t.detach(),
        }

    # ---- evaluation ----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        n = 0
        tot_lm = 0.0
        tot_num = 0.0
        tot_charge_ce = 0.0
        abs_err_log_vo_real = 0.0   # MAE in REAL log_vo units (un-standardized)
        abs_err_log_K_real = 0.0
        abs_err_dEf_real = 0.0
        n_correct = 0
        n_charge = 0

        ns = self.train_ds.numeric_stats
        for batch in self.val_loader:
            out = self._forward_loss(batch)
            B = batch["input_ids"].size(0)
            n += B
            tot_lm += float(out["lm_loss"]) * B
            tot_num += float(out["num_loss"]) * B
            tot_charge_ce += float(out["charge_loss"]) * B

            # un-standardize for the human-readable MAE metric
            pred_log_vo = (
                out["pred_log_vo_std"].cpu().float().numpy() * ns.log_vo_std
                + ns.log_vo_mean
            )
            targ_log_vo = (
                out["targ_log_vo_std"].cpu().float().numpy() * ns.log_vo_std
                + ns.log_vo_mean
            )
            abs_err_log_vo_real += float(np.abs(pred_log_vo - targ_log_vo).sum())

            n_correct += int((out["pred_charge"] == out["targ_charge"]).sum())
            n_charge += B

        self.model.train()
        return {
            "val_lm_loss": tot_lm / max(1, n),
            "val_num_loss": tot_num / max(1, n),
            "val_charge_loss": tot_charge_ce / max(1, n),
            "val_log_vo_mae_real": abs_err_log_vo_real / max(1, n),
            "val_charge_acc": n_correct / max(1, n_charge),
            "val_n": n,
        }

    # ---- cross-check on V55-Ext sputter V_O rows (§5.1.6) -------------------

    @torch.no_grad()
    def cross_check_v55ext(self) -> dict[str, Any]:
        """For each (up to N) sputter V_O ground-truth row in the v58 CSV,
        build the Physics-LLM prompt and ask the numeric prior head for
        log_vo. Check |pred - true| <= 0.5 dex.
        """
        csv_path = Path(self.cfg.cross_check_csv)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        if not csv_path.exists():
            return {"n_rows": 0, "n_hit": 0, "error": f"missing {csv_path}"}

        df = pd.read_csv(csv_path)
        # filter: sputter method, non-null VC, dedup on (spec, atm, T, doi)
        m = df["method"].astype(str).str.lower()
        mask = (
            (m.str.contains("sputter") | m.str.contains("magnetron"))
            & df["vacancy_concentration"].notna()
        )
        sub = df[mask].copy()
        sub = sub.drop_duplicates(
            subset=["dopant_spec", "method", "atmosphere", "temperature_C", "doi"],
            keep="first",
        )
        # take the first N as the canonical V55-Ext sputter VC subset
        sub = sub.head(self.cfg.cross_check_n)

        ns = self.train_ds.numeric_stats
        was_training = self.model.training
        self.model.eval()
        tok = self.model.tokenizer
        rows_out: list[dict] = []
        n_hit = 0
        for _, row in sub.iterrows():
            spec = str(row["dopant_spec"])
            dopant = spec.split(":")[0] if ":" in spec else spec
            if dopant.lower() == "undoped":
                dopant = "undoped"
            try:
                conc_str = spec.split(":", 1)[1] if ":" in spec else "0.0"
                conc = float(conc_str)
            except Exception:
                conc = 0.0
            atm = str(row["atmosphere"])
            try:
                T_C = float(row["temperature_C"])
            except Exception:
                T_C = 800.0
            true_log_vo = float(row["vacancy_concentration"])

            inp = {
                "dopant": dopant,
                "dopant_concentration": conc,
                "atmosphere": atm,
                "temperature_C": T_C,
                "fermi_level_eV": 4.0,
            }
            prompt = build_prompt(inp)
            try:
                ptxt = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                ptxt = prompt
            enc = tok(ptxt, return_tensors="pt", max_length=2048,
                      truncation=True, padding=False)
            ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)

            outputs = self.model.backbone(
                input_ids=ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False,
            )
            lh = outputs.hidden_states[-1]
            mk = attn.unsqueeze(-1).to(lh.dtype)
            pooled = (lh * mk).sum(1) / mk.sum(1).clamp_min(1.0)
            nh = self.model.numeric_head(
                pooled.to(self.model.numeric_head.net[0].weight.dtype)
            )
            pred_std = float(nh["log_vo"].squeeze().item())
            pred_real = pred_std * ns.log_vo_std + ns.log_vo_mean
            err = pred_real - true_log_vo
            hit = abs(err) <= 0.5
            n_hit += int(hit)
            rows_out.append({
                "dopant_spec": spec,
                "atmosphere": atm,
                "temperature_C": T_C,
                "true_log_vo": true_log_vo,
                "pred_log_vo": pred_real,
                "abs_err": abs(err),
                "hit_pm0p5": hit,
                "doi": str(row.get("doi", "")),
            })

        if was_training:
            self.model.train()
        return {
            "n_rows": len(rows_out),
            "n_hit": n_hit,
            "hit_rate": n_hit / max(1, len(rows_out)),
            "mae_log_vo": float(np.mean([r["abs_err"] for r in rows_out])) if rows_out else float("nan"),
            "rows": rows_out,
        }

    # ---- save / load ---------------------------------------------------------

    def save_adapter(self, tag: str) -> Path:
        out_dir = Path(self.cfg.output_dir) / f"adapter-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapter (peft) — backbone is a PeftModel
        try:
            self.model.backbone.save_pretrained(str(out_dir))
        except Exception as e:
            logger.warning("save_pretrained on backbone failed: %s", e)
        # numeric head + z_proj weights (state_dict, fp32) — not part of PEFT
        head_state = {
            "numeric_head": self.model.numeric_head.state_dict(),
            "z_proj": self.model.z_proj.state_dict(),
        }
        torch.save(head_state, out_dir / "heads.pt")
        # numeric stats so the cache builder + Stage-3 can un-standardize
        if self.train_ds.numeric_stats is not None:
            with open(out_dir / "numeric_stats.json", "w") as f:
                json.dump(self.train_ds.numeric_stats.to_dict(), f, indent=2)
        return out_dir

    def _append_metrics(self, payload: dict) -> None:
        self._metrics_log.append(payload)
        with open(self.metrics_path, "w") as f:
            json.dump({
                "run_dir": str(self.cfg.output_dir),
                "config": {
                    "num_epochs": self.cfg.num_epochs,
                    "batch_size": self.cfg.batch_size,
                    "grad_accum_steps": self.cfg.grad_accum_steps,
                    "learning_rate": self.cfg.learning_rate,
                    "lambda_num": self.cfg.lambda_num,
                    "lambda_charge": self.cfg.lambda_charge,
                    "max_steps": self.cfg.max_steps,
                    "bf16": self.cfg.bf16,
                },
                "numeric_stats": (
                    self.train_ds.numeric_stats.to_dict()
                    if self.train_ds.numeric_stats else None
                ),
                "step_loss_log": self._step_loss_log,
                "metrics_log": self._metrics_log,
            }, f, indent=2)

    # ---- train loop ----------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        torch.manual_seed(self.cfg.seed)
        self.model.train()
        accum = 0
        running = {"loss": 0.0, "lm": 0.0, "num": 0.0, "charge": 0.0, "n": 0}
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

                running["loss"] += float(out["loss"])
                running["lm"] += float(out["lm_loss"])
                running["num"] += float(out["num_loss"])
                running["charge"] += float(out["charge_loss"])
                running["n"] += 1
                accum += 1

                if accum >= self.cfg.grad_accum_steps:
                    # clip
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.opt_step += 1
                    accum = 0

                    if self.opt_step % max(1, self.cfg.log_steps) == 0:
                        n = running["n"]
                        avg_loss = running["loss"] / n
                        avg_lm = running["lm"] / n
                        avg_num = running["num"] / n
                        avg_charge = running["charge"] / n
                        lr_now = self.scheduler.get_last_lr()[0]
                        msg = (
                            f"[step {self.opt_step}/{self._total_opt_steps}] "
                            f"loss={avg_loss:.4f} lm={avg_lm:.4f} "
                            f"num={avg_num:.4f} chg={avg_charge:.4f} "
                            f"lr={lr_now:.2e} elapsed={time.time()-t0:.1f}s"
                        )
                        logger.info(msg)
                        print(msg, flush=True)
                        self._step_loss_log.append({
                            "opt_step": self.opt_step,
                            "loss": avg_loss, "lm_loss": avg_lm,
                            "num_loss": avg_num, "charge_loss": avg_charge,
                            "lr": lr_now,
                        })
                        running = {"loss": 0.0, "lm": 0.0, "num": 0.0,
                                   "charge": 0.0, "n": 0}

                    if (
                        self.cfg.eval_steps > 0
                        and self.opt_step % self.cfg.eval_steps == 0
                    ):
                        val_metrics = self.evaluate()
                        val_metrics.update({"opt_step": self.opt_step})
                        self._append_metrics(val_metrics)
                        logger.info("eval @ step %d: %s", self.opt_step, val_metrics)

                    if (
                        self.cfg.save_steps > 0
                        and self.opt_step % self.cfg.save_steps == 0
                    ):
                        path = self.save_adapter(f"step-{self.opt_step}")
                        logger.info("saved adapter to %s", path)

                    if (
                        self.cfg.max_steps is not None
                        and self.opt_step >= self.cfg.max_steps
                    ):
                        break

            if (
                self.cfg.max_steps is not None
                and self.opt_step >= self.cfg.max_steps
            ):
                break

        # final eval + cross-check
        final = self.evaluate()
        cc = self.cross_check_v55ext()
        final["cross_check_v55ext"] = {
            "n_rows": cc["n_rows"],
            "n_hit": cc["n_hit"],
            "hit_rate": cc["hit_rate"],
            "mae_log_vo": cc.get("mae_log_vo"),
        }
        final["opt_step"] = self.opt_step
        final["_final"] = True
        self._append_metrics(final)

        # also dump the per-row cross-check
        cc_path = Path(self.cfg.output_dir) / "cross_check_v55ext.json"
        with open(cc_path, "w") as f:
            json.dump(cc, f, indent=2)

        final_adapter = self.save_adapter(f"final-step-{self.opt_step}")
        logger.info("final adapter: %s", final_adapter)
        return {
            "final_metrics": final,
            "cross_check": cc,
            "final_adapter_dir": str(final_adapter),
            "metrics_path": str(self.metrics_path),
        }
