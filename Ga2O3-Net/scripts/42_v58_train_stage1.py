"""
Phase-58 Stage-1 Physics-LLM trainer entry point.

Loads `PhysicsLLM.from_pretrained_qlora` (Qwen2.5-1.5B + NF4 + LoRA r=8),
builds train + val Stage1QADatasets from
`data/processed/stage1_qa_pairs[_val].jsonl`, runs `Stage1Trainer.fit()`,
and writes:
  - <out>/metrics.json                 — train+val loss + numeric MAE + charge acc
  - <out>/cross_check_v55ext.json      — per-row V55-Ext sputter V_O check (§5.1.6)
  - <out>/adapter-step-<N>/            — saved LoRA adapters

Usage:
    # smoke (this task)
    python scripts/42_v58_train_stage1.py --max-steps 20 --gpu 1

    # full 3-epoch run (next orchestrator step)
    python scripts/42_v58_train_stage1.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# project root
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="config/stage1_physics_llm.yaml")
    p.add_argument("--data", type=str,
                   default="data/processed/stage1_qa_pairs.jsonl")
    p.add_argument("--val-data", type=str,
                   default="data/processed/stage1_qa_pairs_val.jsonl")
    p.add_argument("--gpu", type=int, default=1,
                   help="CUDA device index (task spec: smoke uses GPU 1)")
    p.add_argument("--output", type=str,
                   default="checkpoints/qwen25_1.5b_v58_physics")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override total optimizer steps (smoke=20)")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="disable gradient checkpointing (more VRAM but faster; "
                        "GPU 1 has 21GB headroom at batch=1)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="per-step micro-batch (doc default 1); raise for "
                        "better GPU utilization. Effective batch stays "
                        "batch_size * grad_accum")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="grad accumulation steps (doc default 16); keep "
                        "batch_size * grad_accum == 16 to preserve effective "
                        "batch from §5.1.5")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # GPU pinning BEFORE importing torch/transformers so device_map="cuda:0"
    # within the visible scope means the requested physical GPU.
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import yaml

    from src.data.stage1_qa_dataset import (
        Stage1QADataset, compute_numeric_stats,
    )
    from src.models.llm_physics import PhysicsLLM
    from src.training.stage1_trainer import Stage1Cfg, Stage1Trainer

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
    log = logging.getLogger("stage1")
    log.info("args: %s", vars(args))

    # config yaml -> cfg overrides
    with open(args.config, "r") as f:
        ycfg = yaml.safe_load(f) or {}
    tcfg = ycfg.get("training", {})

    cfg = Stage1Cfg(
        output_dir=out_dir,
        num_epochs=args.num_epochs if args.num_epochs is not None
                   else int(tcfg.get("num_epochs", 3)),
        batch_size=args.batch_size if args.batch_size is not None
                   else int(tcfg.get("batch_size", 1)),
        grad_accum_steps=args.grad_accum if args.grad_accum is not None
                         else int(tcfg.get("gradient_accumulation_steps", 16)),
        learning_rate=args.lr if args.lr is not None
                      else float(tcfg.get("learning_rate", 2.0e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        eval_steps=args.eval_steps if args.eval_steps is not None
                   else int(tcfg.get("eval_steps", 200)),
        save_steps=args.save_steps if args.save_steps is not None
                   else int(tcfg.get("save_steps", 400)),
        log_steps=args.log_steps,
        max_steps=args.max_steps,
        lambda_num=1.0,
        lambda_charge=0.3,
        bf16=bool(tcfg.get("bf16", True)),
        seed=args.seed,
        num_workers=0,
    )
    log.info("Stage1Cfg: %s", cfg)

    # Optional Liger kernel speed-up (Qwen2 family) — pre-import patches the
    # transformers modules so subsequent from_pretrained gets fused RMSNorm etc.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()
        log.info("liger_kernel patched Qwen2 (fused RMSNorm, swiglu, RoPE)")
    except Exception as e:
        log.info("liger_kernel not applied (%s) — using stock kernels", e)

    # Build the real QLoRA model.  CUDA_VISIBLE_DEVICES pins us to one GPU
    # so device_map="cuda:0" here = physical GPU args.gpu.
    log.info("loading PhysicsLLM (Qwen2.5-1.5B NF4 + LoRA r=8) on cuda:0 ...")
    t0 = time.time()
    model = PhysicsLLM.from_pretrained_qlora(
        device_map={"": "cuda:0"},
        max_len=2048,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    log.info("model loaded in %.1fs", time.time() - t0)
    log.info("trainable breakdown: %s", model.n_trainable_breakdown())

    # Move the small fp32 heads (z_proj, numeric_head) to GPU; the 4-bit
    # backbone is already placed by from_pretrained.
    dev = torch.device("cuda:0")
    model.z_proj.to(dev)
    model.numeric_head.to(dev)
    # cast heads to bf16 to match the autocast path (avoids per-step dtype casts)
    model.numeric_head.to(dtype=torch.bfloat16)
    model.z_proj.to(dtype=torch.bfloat16)

    # numeric stats from TRAIN ONLY (no leak)
    log.info("computing numeric stats on TRAIN file: %s", args.data)
    stats = compute_numeric_stats(args.data)
    log.info("numeric stats: %s", stats.to_dict())

    # floor-pile-up histogram (memo: ~21 % at log_vo=5.0)
    import json as _json
    floor_n = 0
    total_n = 0
    with open(args.data) as f:
        for line in f:
            r = _json.loads(line)
            total_n += 1
            if abs(float(r["label"]["log_vo_predicted"]) - 5.0) < 1e-6:
                floor_n += 1
    log.info(
        "floor-pile-up (log_vo == 5.0): %d/%d = %.1f%%",
        floor_n, total_n, 100.0 * floor_n / max(1, total_n),
    )

    train_ds = Stage1QADataset(args.data, model.tokenizer,
                               numeric_stats=stats, max_len=2048)
    val_ds = Stage1QADataset(args.val_data, model.tokenizer,
                             numeric_stats=stats, max_len=2048)
    log.info("train rows: %d ; val rows: %d", len(train_ds), len(val_ds))

    trainer = Stage1Trainer(model=model, train_ds=train_ds, val_ds=val_ds, cfg=cfg)
    log.info(
        "total optimizer steps planned: %d (warmup %d)",
        trainer._total_opt_steps, trainer._warmup,
    )

    result = trainer.fit()
    log.info("DONE.  metrics: %s", result["metrics_path"])
    log.info("final adapter: %s", result["final_adapter_dir"])

    summary = {
        "ok": True,
        "metrics_path": result["metrics_path"],
        "final_adapter_dir": result["final_adapter_dir"],
        "final_metrics": result["final_metrics"],
        "cross_check_n_rows": result["cross_check"]["n_rows"],
        "cross_check_n_hit": result["cross_check"]["n_hit"],
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
