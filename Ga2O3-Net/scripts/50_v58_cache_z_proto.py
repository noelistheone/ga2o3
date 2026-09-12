"""
Phase-58 LLMBridge refinement — Step 1/3: cache z_proto per paper.

Loads the FROZEN Stage-2 ProcessLLM (Qwen2.5-7B NF4 + LoRA, the
`adapter-final-step-500` adapter + its `heads_bridge.pt` z_proj) and runs
`extract_z_proto` ONCE per paper that has a Methods .txt, writing

    data/processed/z_proto_cache_v58.npz

keyed by paper_id (the markdown/methods file hash) -> 32-d fp16 vector.

Why this is the efficient path (per task spec + feedback_gpu_over_cpu):
the 7B is FROZEN; we only forward each Methods text once (masked mean-pool,
NO autoregressive generate), so z_proto for ~550 papers is a few hundred
forwards (~5-10 min on one RTX 3090). The 25k bridge is then trained on the
cached (z_proto, z_phys) pairs (script 51) without ever touching the 7B again.

GPU: GPU 1 ONLY. CUDA_VISIBLE_DEVICES is pinned to the requested device
BEFORE torch is imported, so "cuda:0" inside this process == physical GPU 1.
GPU 0 (an unrelated training run) is never referenced.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage2-adapter", type=str,
        default="checkpoints/qwen25_7b_v58_process_cont/adapter-final-step-500",
        help="Stage-2 ProcessLLM adapter dir (contains adapter_model.safetensors "
             "+ heads_bridge.pt with the trained z_proj).")
    p.add_argument("--labels-jsonl", type=str,
                   default="data/processed/paper_protocol_labels.jsonl",
                   help="paper_id -> methods_text_path mapping (script 46 output).")
    p.add_argument("--methods-dir", type=str,
                   default="data/processed/paper_methods")
    p.add_argument("--out", type=str,
                   default="data/processed/z_proto_cache_v58.npz")
    p.add_argument("--gpu", type=int, default=1,
                   help="Physical GPU index (task: GPU 1 ONLY). -1 = CPU.")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0,
                   help="debug: cap number of papers (0 = all).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Pin GPU BEFORE importing torch so device_map="cuda:0" == physical --gpu.
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import numpy as np
    from peft import PeftModel
    from src.models.llm_process import ProcessLLM

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("50_z_proto")

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda:0" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    log.info("device: %s (physical GPU %s)", dev, args.gpu)

    # 1. paper_id -> methods .txt path
    labels_jsonl = HERE / args.labels_jsonl
    methods_dir = HERE / args.methods_dir
    paper_methods: dict[str, Path] = {}
    with open(labels_jsonl, "r") as f:
        for line in f:
            r = json.loads(line)
            pid = r["paper_id"]
            mp = r.get("methods_text_path")
            mpath = (HERE / mp) if mp else (methods_dir / f"{pid}.txt")
            if mpath.exists():
                paper_methods[pid] = mpath
    log.info("papers with a Methods .txt: %d", len(paper_methods))
    if args.limit:
        paper_methods = dict(sorted(paper_methods.items())[: args.limit])
        log.info("--limit applied: %d papers", len(paper_methods))

    # 2. Load FROZEN ProcessLLM (Stage-2 adapter) + restore trained z_proj
    adapter_dir = HERE / args.stage2_adapter
    log.info("loading frozen ProcessLLM (Qwen2.5-7B NF4 + LoRA) ...")
    t0 = time.time()
    process_llm = ProcessLLM.from_pretrained_qlora(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device_map={"": "cuda:0"} if dev.type == "cuda" else None,
        max_len=args.max_seq_len,
        gradient_checkpointing=False,
        init_adapter_dir=str(adapter_dir),   # load the trained LoRA adapter
    )
    log.info("ProcessLLM loaded in %.1fs", time.time() - t0)

    heads_path = adapter_dir / "heads_bridge.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"heads_bridge.pt missing at {heads_path}")
    hb = torch.load(heads_path, map_location="cpu")
    if "z_proj" not in hb:
        raise KeyError("heads_bridge.pt has no 'z_proj' state_dict")
    process_llm.z_proj.load_state_dict(hb["z_proj"])
    log.info("restored trained z_proj from heads_bridge.pt")

    # Keep z_proj in fp32 on-device. The NF4 backbone (+ prepare_model_for_kbit_
    # training) emits an fp32 last-hidden-state, so a bf16 z_proj would mismatch
    # ("mat1 float != mat2 BFloat16"). fp32 z_proj on fp32 pooled is consistent;
    # backbone matmuls still run in bf16 via autocast below.
    process_llm.z_proj.to(dev, dtype=torch.float32)
    process_llm.freeze_for_stage3()
    log.info("ProcessLLM frozen, n_trainable=%d", process_llm.n_trainable())

    # 3. One forward per paper -> z_proto (mean-pool, no generate)
    pids = sorted(paper_methods.keys())
    keys_out: list[str] = []
    z_out: list[np.ndarray] = []
    n_fail = 0
    util_samples: list[int] = []
    t1 = time.time()
    for i, pid in enumerate(pids):
        try:
            text = paper_methods[pid].read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                n_fail += 1
                continue
            with torch.no_grad():
                if dev.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = process_llm.extract_z_proto([text])   # [1, 32]
                else:
                    z = process_llm.extract_z_proto([text])
            z = z.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()
            if not np.isfinite(z).all():
                log.warning("non-finite z_proto for %s — skipping", pid)
                n_fail += 1
                continue
            keys_out.append(pid)
            z_out.append(z.astype(np.float16))
        except Exception as e:
            log.warning("z_proto failed for %s: %s", pid, e)
            n_fail += 1
            continue

        if (i + 1) % 25 == 0 or (i + 1) == len(pids):
            rate = (i + 1) / max(1e-9, time.time() - t1)
            log.info("  z_proto %d/%d  (%.2f papers/s)", i + 1, len(pids), rate)
            if dev.type == "cuda":
                try:
                    import subprocess
                    u = subprocess.check_output(
                        ["nvidia-smi", "--id=" + str(args.gpu),
                         "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"],
                        text=True).strip()
                    util_gpu = int(u.split(",")[0])
                    util_samples.append(util_gpu)
                    log.info("  [GPU %s] util=%s%% mem=%s MiB", args.gpu,
                             u.split(",")[0].strip(), u.split(",")[1].strip())
                except Exception:
                    pass

    if not z_out:
        raise RuntimeError("no z_proto vectors computed")

    feats = np.stack(z_out, axis=0).astype(np.float16)   # [N, 32]
    keys_arr = np.array(keys_out, dtype="<U32")

    meta = {
        "n_papers": int(len(keys_out)),
        "n_failed": int(n_fail),
        "z_dim": 32,
        "dtype": "float16",
        "stage2_adapter": str(args.stage2_adapter),
        "max_seq_len": args.max_seq_len,
        "pooling": "masked_mean (no autoregressive generate)",
        "key_scheme": "paper_id (methods/markdown file hash)",
        "gpu_util_mean_pct": (float(sum(util_samples)) / len(util_samples))
                             if util_samples else None,
        "gpu_util_max_pct": (max(util_samples) if util_samples else None),
    }
    # Save keyed-by-name (matches z_phys cache style) AND as keys/features arrays
    # so script 51 / 52 can load either way.
    save_kwargs = {pid: z for pid, z in zip(keys_out, z_out)}
    save_kwargs["keys"] = keys_arr
    save_kwargs["features"] = feats
    save_kwargs["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez(out_path, **save_kwargs)
    log.info("wrote %s  (N=%d, dim=32, fp16)", out_path, len(keys_out))

    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "n_papers": len(keys_out),
        "n_failed": n_fail,
        "gpu_util_mean_pct": meta["gpu_util_mean_pct"],
        "gpu_util_max_pct": meta["gpu_util_max_pct"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
