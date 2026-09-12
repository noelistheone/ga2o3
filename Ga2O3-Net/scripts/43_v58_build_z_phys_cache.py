"""
Phase-58 z_phys cache builder.

Loads the Stage-1 trained Physics-LLM, iterates over the (dopant, atm, T)
combinations enumerated in `data/processed/dft_chemical_potentials.json`,
and saves `data/processed/z_phys_cache_v58.npz` keyed by
"<dopant>|<atm>|<T_C>" -> 32-d float16 z_phys vector (design doc §3.6.5).

This is the mean-pooled embedding path (no autoregressive generate) — one
forward per condition, on GPU 1 only, per the project memory rule
`feedback_gpu_over_cpu`.

Usage:
    python scripts/43_v58_build_z_phys_cache.py \\
        --adapter-dir checkpoints/qwen25_1.5b_v58_physics/adapter-final-step-XXX \\
        --gpu 1
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
    p.add_argument("--adapter-dir", type=str, required=True,
                   help="path to a Stage-1 saved adapter directory "
                        "(contains adapter_config.json + heads.pt + "
                        "numeric_stats.json)")
    p.add_argument("--dft-json", type=str,
                   default="data/processed/dft_chemical_potentials.json")
    p.add_argument("--output", type=str,
                   default="data/processed/z_phys_cache_v58.npz")
    p.add_argument("--gpu", type=int, default=1,
                   help="CUDA device index (task spec: GPU 1)")
    p.add_argument("--default-conc", type=float, default=0.02,
                   help="placeholder atomic-fraction for prompts (V_O is a "
                        "weak function of concentration in this design)")
    return p.parse_args()


_ATM_LABEL = {
    "Ar": "Ar",
    "Ar_O2_4_1": "Ar:O2=4:1",
    "Ar_O2_1_1": "Ar:O2=1:1",
    "O2": "O2",
}


def main() -> int:
    args = parse_args()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from peft import PeftModel

    from src.data.stage1_qa_dataset import build_prompt
    from src.models.llm_physics import PhysicsLLM

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("z_phys_cache")

    # ---- 1. load base model + apply Stage-1 LoRA adapter -------------------
    log.info("loading base PhysicsLLM (Qwen2.5-1.5B NF4) on cuda:0 ...")
    t0 = time.time()
    base = PhysicsLLM.from_pretrained_qlora(
        device_map={"": "cuda:0"},
        max_len=2048,
        gradient_checkpointing=False,
    )
    log.info("base loaded in %.1fs", time.time() - t0)

    # The base ALREADY wraps Qwen2.5-1.5B in a fresh PEFT model.  To load the
    # Stage-1 adapter we re-wrap the underlying transformer with PeftModel.
    adapter_dir = Path(args.adapter_dir)
    log.info("loading LoRA adapter from %s", adapter_dir)
    underlying = base.backbone.get_base_model()  # strip the freshly-init LoRA
    new_peft = PeftModel.from_pretrained(underlying, str(adapter_dir))
    base.backbone = new_peft

    # restore the trained head weights
    heads_pt = adapter_dir / "heads.pt"
    if heads_pt.exists():
        state = torch.load(heads_pt, map_location="cuda:0")
        if "numeric_head" in state:
            base.numeric_head.load_state_dict(state["numeric_head"])
        if "z_proj" in state:
            base.z_proj.load_state_dict(state["z_proj"])
        log.info("loaded heads.pt")
    else:
        log.warning("heads.pt not found in %s — using freshly-initialised heads",
                    adapter_dir)

    base.z_proj.to("cuda:0", dtype=torch.bfloat16)
    base.numeric_head.to("cuda:0", dtype=torch.bfloat16)
    base.freeze_for_stage3()  # eval mode + no grad
    log.info("model ready (eval, frozen)")

    # ---- 2. enumerate (dopant, atm, T) combos -------------------------------
    with open(args.dft_json) as f:
        dft = json.load(f)
    entries = dft["entries"]
    log.info("dft_chemical_potentials: %d dopants", len(entries))

    combos: list[tuple[str, str, int]] = []
    for el, info in entries.items():
        for at_T in info["by_atm_T"]:
            atm, T = at_T.rsplit("|", 1)
            # T is like "700C"
            T_C = int(T.replace("C", ""))
            combos.append((el, atm, T_C))
    log.info("total (dopant, atm, T_C) combos: %d", len(combos))

    # ---- 3. one forward per combo, mean-pool, project 1536->32 -------------
    out_dict: dict[str, np.ndarray] = {}
    tok = base.tokenizer
    dev = torch.device("cuda:0")
    t1 = time.time()
    with torch.no_grad():
        for i, (el, atm_key, T_C) in enumerate(combos):
            info = entries[el]
            ef = float(info.get("fermi_level_eV", 4.0))
            atm_label = _ATM_LABEL.get(atm_key, atm_key)
            inp = {
                "dopant": el,
                "dopant_concentration": args.default_conc,
                "atmosphere": atm_label,
                "temperature_C": float(T_C),
                "fermi_level_eV": ef,
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
            ids = enc["input_ids"].to(dev)
            attn = enc["attention_mask"].to(dev)

            outputs = base.backbone(
                input_ids=ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False,
            )
            lh = outputs.hidden_states[-1]
            mk = attn.unsqueeze(-1).to(lh.dtype)
            pooled = (lh * mk).sum(1) / mk.sum(1).clamp_min(1.0)   # [1,H]
            z = base.z_proj(pooled.to(base.z_proj.weight.dtype))    # [1,32]
            key = f"{el}|{atm_key}|{T_C}"
            out_dict[key] = z.squeeze(0).cpu().to(torch.float16).numpy()
            if (i + 1) % 50 == 0 or i + 1 == len(combos):
                log.info("  [%d/%d] %.1fs elapsed", i + 1, len(combos),
                         time.time() - t1)

    # ---- 4. save ------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(out_path, **out_dict)
    log.info("wrote %s (%d keys)", out_path, len(out_dict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
