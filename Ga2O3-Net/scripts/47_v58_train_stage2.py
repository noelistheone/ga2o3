"""
Phase-58 Stage-2 Process-LLM + Bridge trainer entry point (§5.2).

Loads:
  - ProcessLLM.from_pretrained_qlora(...)        # Qwen2.5-7B + NF4 + LoRA r=16
  - PhysicsLLM frozen (Stage-1 adapter if present; else None → random z_phys)
  - LLMBridge (2 layers, dim=32)

Builds train + val Stage2PaperDataset from `data/processed/paper_protocol_labels.jsonl`,
splits 9:1, runs `Stage2Trainer.fit()`, and writes:
  - <out>/metrics.json
  - <out>/train.log
  - <out>/adapter-step-<N>/                      # LoRA adapter + heads_bridge.pt
  - <out>/run_summary.json

Usage:
    # smoke (orchestrator-approved; only after GPU 1 is confirmed free)
    python scripts/47_v58_train_stage2.py --max-steps 20 --gpu 1

    # full 2-epoch run (production)
    python scripts/47_v58_train_stage2.py --gpu 1

Hardware: per §5.2.4 / hardware block, the real run targets GPU 1.  GPU pinning
happens via CUDA_VISIBLE_DEVICES BEFORE torch is imported, so within the script
"cuda:0" always means the physical `--gpu` device.

VRAM expectation (Qwen2.5-7B NF4 + LoRA r=16 + grad-checkpointing, batch=1,
seq=4096): roughly 12-14 GiB activations + 4-5 GiB weights ≈ 16-18 GiB on the
process-LLM device, plus ~3 GiB if the frozen Physics-LLM is co-located.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="config/stage2_process_llm.yaml")
    p.add_argument("--labels-jsonl", type=str,
                   default="data/processed/paper_protocol_labels.jsonl")
    p.add_argument("--gpu", type=int, default=1,
                   help="CUDA device index (task: full run on GPU 1; "
                        "use -1 for CPU.")
    p.add_argument("--output", type=str,
                   default="checkpoints/qwen25_7b_v58_process")
    p.add_argument("--stage1-adapter", type=str, default=None,
                   help="Optional path to Stage-1 PhysicsLLM adapter dir. "
                        "If absent the trainer uses random z_phys for the "
                        "bridge MSE term (smoke path).")
    p.add_argument("--init-adapter", type=str, default=None,
                   help="Resume: load ProcessLLM LoRA weights from this adapter dir "
                        "(crash-resilience / continue training).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override total optimizer steps (smoke=20)")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Pin GPU BEFORE importing torch / transformers so device_map="cuda:0"
    # within the visible scope means the requested physical GPU.
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import yaml

    from src.data.stage2_paper_dataset import Stage2PaperDataset
    from src.models.llm_process import ProcessLLM
    from src.models.llm_bridge import LLMBridge
    from src.training.stage2_trainer import Stage2Cfg, Stage2Trainer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("stage2")
    log.info("args: %s", vars(args))

    with open(args.config, "r") as f:
        ycfg = yaml.safe_load(f) or {}
    tcfg = ycfg.get("training", {})

    cfg = Stage2Cfg(
        output_dir=out_dir,
        num_epochs=args.num_epochs if args.num_epochs is not None
                   else int(tcfg.get("num_epochs", 2)),
        batch_size=args.batch_size if args.batch_size is not None
                   else int(tcfg.get("batch_size", 1)),
        grad_accum_steps=args.grad_accum if args.grad_accum is not None
                         else int(tcfg.get("gradient_accumulation_steps", 16)),
        learning_rate=args.lr if args.lr is not None
                      else float(tcfg.get("learning_rate", 1.5e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        max_seq_length=args.max_seq_length if args.max_seq_length is not None
                       else int(tcfg.get("max_seq_length", 4096)),
        eval_steps=args.eval_steps if args.eval_steps is not None
                   else int(tcfg.get("eval_steps", 200)),
        save_steps=args.save_steps if args.save_steps is not None
                   else int(tcfg.get("save_steps", 400)),
        log_steps=args.log_steps,
        max_steps=args.max_steps,
        lambda_mlm=1.0,
        lambda_nce=0.5,
        lambda_protocol=0.3,
        lambda_bridge=0.2,
        nce_temperature=0.07,
        bf16=bool(tcfg.get("bf16", True)),
        seed=args.seed,
        num_workers=0,
    )
    log.info("Stage2Cfg: %s", cfg)

    # Optional Liger kernel patch (Qwen2 family)
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()
        log.info("liger_kernel patched Qwen2")
    except Exception as e:
        log.info("liger_kernel not applied (%s) — stock kernels", e)

    # Build ProcessLLM (real QLoRA Qwen2.5-7B)
    log.info("loading ProcessLLM (Qwen2.5-7B NF4 + LoRA r=16) on cuda:0 ...")
    t0 = time.time()
    process_llm = ProcessLLM.from_pretrained_qlora(
        device_map={"": "cuda:0"},
        max_len=cfg.max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        init_adapter_dir=args.init_adapter,
    )
    log.info("ProcessLLM loaded in %.1fs (init_adapter=%s)", time.time() - t0,
             args.init_adapter)
    log.info("trainable breakdown: %s", process_llm.n_trainable_breakdown())

    dev = torch.device("cuda:0")
    # Resume: restore the trained z_proj + protocol_head from the adapter dir's heads file.
    if args.init_adapter:
        hb = os.path.join(args.init_adapter, "heads_bridge.pt")
        try:
            if os.path.exists(hb) and os.path.getsize(hb) > 0:
                st = torch.load(hb, map_location="cpu")
                if "z_proj" in st:
                    process_llm.z_proj.load_state_dict(st["z_proj"]); log.info("restored z_proj")
                if "protocol_head" in st:
                    process_llm.protocol_head.load_state_dict(st["protocol_head"]); log.info("restored protocol_head")
            else:
                log.warning("heads_bridge.pt missing/empty (%s) — heads re-init (LoRA backbone preserved)", hb)
        except Exception as e:
            log.warning("heads_bridge.pt unreadable (%s: %s) — heads re-init (LoRA preserved)", type(e).__name__, e)
    # Move + cast the fp32 heads to match autocast bf16
    process_llm.z_proj.to(dev, dtype=torch.bfloat16)
    process_llm.protocol_head.to(dev, dtype=torch.bfloat16)

    # Build LLMBridge (trained)
    bridge = LLMBridge(dim=32, n_heads=4, n_layers=2, dropout=0.1)
    bridge.to(dev, dtype=torch.bfloat16)

    # Optionally load Stage-1 PhysicsLLM (frozen)
    physics_llm = None
    if args.stage1_adapter:
        log.info("loading frozen PhysicsLLM with stage-1 adapter at %s",
                 args.stage1_adapter)
        try:
            from src.models.llm_physics import PhysicsLLM
            from peft import PeftModel
            # The PhysicsLLM is loaded on the SAME visible device (cuda:0).
            # For the production run §5.2.4 puts it on GPU 1 *separately*; with
            # CUDA_VISIBLE_DEVICES=1 it co-locates on the only visible GPU.
            physics_llm = PhysicsLLM.from_pretrained_qlora(
                device_map={"": "cuda:0"},
                max_len=2048,
                gradient_checkpointing=False,
            )
            physics_llm.backbone = PeftModel.from_pretrained(
                physics_llm.backbone, args.stage1_adapter,
            )
            physics_llm.freeze_for_stage3()
            log.info("PhysicsLLM loaded + frozen")
        except Exception as e:
            log.warning("Failed to load Stage-1 PhysicsLLM (%s) — falling back "
                        "to random z_phys for bridge MSE", e)
            physics_llm = None
    else:
        log.info("No --stage1-adapter; using random z_phys (smoke path).")

    # Build datasets (9:1 split, deterministic shuffle)
    full_ds = Stage2PaperDataset(args.labels_jsonl)
    n = len(full_ds)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(n * args.val_frac))
    val_idx = set(idx[:n_val])
    # Build via in-place slicing of samples (preserves repo_root etc.)
    train_samples = [s for i, s in enumerate(full_ds.samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(full_ds.samples) if i in val_idx]
    full_ds.samples = train_samples
    full_ds._paper_to_idx = {s.paper_id: i for i, s in enumerate(train_samples)}

    # build a sibling val dataset object
    val_ds = Stage2PaperDataset.__new__(Stage2PaperDataset)
    val_ds.path = full_ds.path
    val_ds.repo_root = full_ds.repo_root
    val_ds.max_chars = full_ds.max_chars
    val_ds.samples = val_samples
    val_ds._paper_to_idx = {s.paper_id: i for i, s in enumerate(val_samples)}

    log.info("train rows: %d ; val rows: %d", len(full_ds), len(val_ds))
    log.info("positive-pair count (train): %d", full_ds.n_positive_pairs())
    log.info("label positive counts (train): %s", full_ds.label_positive_counts())

    trainer = Stage2Trainer(
        process_llm=process_llm,
        physics_llm_frozen=physics_llm,
        bridge=bridge,
        train_ds=full_ds, val_ds=val_ds,
        cfg=cfg,
    )
    log.info("total opt steps planned: %d (warmup %d)",
             trainer._total_opt_steps, trainer._warmup)

    result = trainer.fit()
    summary = {
        "ok": True,
        "metrics_path": result["metrics_path"],
        "final_adapter_dir": result["final_adapter_dir"],
        "final_metrics": result["final_metrics"],
        "n_train_papers": len(full_ds),
        "n_val_papers": len(val_ds),
        "stage1_adapter_used": bool(args.stage1_adapter and physics_llm is not None),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
