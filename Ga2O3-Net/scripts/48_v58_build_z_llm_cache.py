"""
Phase-58 Stage-3 prerequisite: build the z_LLM cache used by V58 fusion.

After BOTH Stages 1 (PhysicsLLM) and 2 (ProcessLLM + Bridge) are trained:

  - PhysicsLLM (frozen, Stage-1 adapter)             -> z_phys [32]
  - ProcessLLM (frozen, Stage-2 adapter + z_proj)   -> z_proto [32]
  - LLMBridge (frozen, Stage-2)                     -> z_LLM = Bridge(z_proto, z_phys)

This script iterates over the union of (paper_id × dopant × atm × T) tuples
required by the V58 fusion model and writes
`data/processed/z_llm_cache_v58.npz` keyed on the SAME `{dopant}|{atm}|{T_C}`
scheme used by `Ga2O3Net._v58_dopant_atm_T_key` (see src/models/ga2o3_net.py
lines ~895-910).

KEY-SCHEME DECISION (documented per task spec)
----------------------------------------------
The doc §3.7.3 floats "per (paper_id, dopant, atm, T) combination, 830 keys"
but the V58 FiLM model uses ONLY `{cation}|{atm_bucket}|{T_bin}` (no paper_id).
We honor the model's existing scheme — the cache is therefore keyed without
paper_id; per-paper variance is *averaged out* at cache build time by
averaging z_LLM over the set of papers that exhibit that (dopant, atm, T)
tuple.  This keeps Stage-3 inference unchanged (`_lookup_z_llm` consumes
exactly `_v58_dopant_atm_T_key`).

If a future V58 revision adds paper_id to the inference key, this script is
the one place that needs updating — add `paper_id` to the key here AND in
`Ga2O3Net._v58_dopant_atm_T_key`.

Pure-CPU bookkeeping for everything except the LoRA forwards.  Requires both
adapters; will refuse to run otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ── key scheme (matches Ga2O3Net._v58_dopant_atm_T_key) ──────────────────────

_V58_ATM_O2FRAC = {"Ar": 0.0, "Ar_O2_4_1": 0.2, "Ar_O2_1_1": 0.5, "O2": 1.0}
_V58_T_BINS = [500, 700, 900, 1100]


def _bin_atmosphere(atm_str: str) -> str | None:
    """Map an experimental atmosphere string to one of the four V58 buckets."""
    s = (atm_str or "").lower().replace(" ", "")
    if "ar" in s and ("o2" in s or "o_2" in s or "oxygen" in s):
        if "1:1" in s or "1/1" in s or "50%" in s:
            return "Ar_O2_1_1"
        if "4:1" in s or "4/1" in s or "20%" in s:
            return "Ar_O2_4_1"
        return "Ar_O2_4_1"
    if "ar" in s:
        return "Ar"
    if "o2" in s or "oxygen" in s or "o_2" in s or "air" in s:
        return "O2"
    if "n2" in s or "nitrogen" in s:
        # No V58 bucket for N2 — closest is Ar (inert).  We default to Ar.
        return "Ar"
    return None


def _bin_temperature(T_C: float) -> int:
    return min(_V58_T_BINS, key=lambda t: abs(t - T_C))


def _bin_dopant(spec_str: str) -> str | None:
    """Extract the primary cation from a DopantSpec string."""
    try:
        from src.data.dopant_spec import DopantSpec
        spec = DopantSpec.parse(str(spec_str).strip())
        if spec is None or getattr(spec, "is_undoped", False) or not spec.components:
            return None
        return max(spec.components, key=lambda c: float(c.conc)).cation
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-adapter", type=str, required=True,
                   help="path to Stage-1 PhysicsLLM adapter dir")
    p.add_argument("--stage2-adapter", type=str, required=True,
                   help="path to Stage-2 ProcessLLM adapter dir (also "
                        "contains heads_bridge.pt with z_proj + protocol_head "
                        "+ bridge state_dicts)")
    p.add_argument("--v58-csv", type=str,
                   default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv")
    p.add_argument("--paper-labels-jsonl", type=str,
                   default="data/processed/paper_protocol_labels.jsonl",
                   help="paper_id -> methods text mapping (script 46 output)")
    p.add_argument("--methods-dir", type=str,
                   default="data/processed/paper_methods")
    p.add_argument("--out", type=str,
                   default="data/processed/z_llm_cache_v58.npz")
    p.add_argument("--gpu", type=int, default=1,
                   help="GPU for the frozen LLM forwards; -1 for CPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import numpy as np
    import pandas as pd

    from src.models.llm_process import ProcessLLM
    from src.models.llm_physics import PhysicsLLM
    from src.models.llm_bridge import LLMBridge

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("48_z_llm_cache")

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load V58 CSV — get (paper_id, dopant, atm, T) tuples
    v58_csv = HERE / args.v58_csv
    if not v58_csv.exists():
        raise FileNotFoundError(f"v58 CSV missing: {v58_csv}")
    df = pd.read_csv(v58_csv)
    log.info("v58 CSV: %d rows", len(df))

    # 2. Load paper -> methods text map
    paper_methods_path: dict[str, Path] = {}
    labels_jsonl = HERE / args.paper_labels_jsonl
    with open(labels_jsonl, "r") as f:
        for line in f:
            r = json.loads(line)
            paper_methods_path[r["paper_id"]] = HERE / r["methods_text_path"]
    log.info("paper_methods labeled: %d", len(paper_methods_path))

    # 3. Enumerate the unique V58-keyed combinations needed by the model,
    #    PER paper_id (we average over papers within the same key).
    needed_per_key_papers: dict[str, set] = defaultdict(set)
    paper_to_keys: dict[str, set] = defaultdict(set)
    skipped_rows = 0
    for _, row in df.iterrows():
        spec = str(row.get("dopant_spec", ""))
        atm_raw = str(row.get("atmosphere", ""))
        try:
            T_C = float(row.get("temperature_C", 800.0))
        except Exception:
            T_C = 800.0
        pid = str(row.get("paper_id", "")) or None

        cation = _bin_dopant(spec)
        atm = _bin_atmosphere(atm_raw)
        if cation is None or atm is None:
            skipped_rows += 1
            continue
        T_bin = _bin_temperature(T_C)
        key = f"{cation}|{atm}|{T_bin}"
        if pid and pid in paper_methods_path:
            needed_per_key_papers[key].add(pid)
            paper_to_keys[pid].add(key)
    log.info("unique V58 keys requested: %d (skipped rows: %d)",
             len(needed_per_key_papers), skipped_rows)

    # 4. Load the frozen Stage-1 PhysicsLLM + Stage-2 ProcessLLM + Bridge
    from peft import PeftModel
    dev = torch.device("cuda:0" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    log.info("device: %s", dev)

    log.info("loading frozen PhysicsLLM (Stage-1 adapter) ...")
    physics_llm = PhysicsLLM.from_pretrained_qlora(
        device_map={"": "cuda:0"} if dev.type == "cuda" else None,
        max_len=2048, gradient_checkpointing=False,
    )
    physics_llm.backbone = PeftModel.from_pretrained(
        physics_llm.backbone, args.stage1_adapter,
    )
    # Load Stage-1 heads
    stage1_heads_path = Path(args.stage1_adapter) / "heads.pt"
    if stage1_heads_path.exists():
        h1 = torch.load(stage1_heads_path, map_location="cpu")
        if "z_proj" in h1:
            physics_llm.z_proj.load_state_dict(h1["z_proj"])
        if "numeric_head" in h1:
            physics_llm.numeric_head.load_state_dict(h1["numeric_head"])
    physics_llm.z_proj.to(dev, dtype=torch.bfloat16)
    physics_llm.freeze_for_stage3()
    log.info("PhysicsLLM ready, n_trainable=%d", physics_llm.n_trainable())

    log.info("loading frozen ProcessLLM (Stage-2 adapter) ...")
    process_llm = ProcessLLM.from_pretrained_qlora(
        device_map={"": "cuda:0"} if dev.type == "cuda" else None,
        max_len=4096, gradient_checkpointing=False,
    )
    process_llm.backbone = PeftModel.from_pretrained(
        process_llm.backbone, args.stage2_adapter,
    )
    # Load Stage-2 heads + bridge
    s2_heads = Path(args.stage2_adapter) / "heads_bridge.pt"
    bridge = LLMBridge(dim=32, n_heads=4, n_layers=2, dropout=0.0)
    if s2_heads.exists():
        h2 = torch.load(s2_heads, map_location="cpu")
        if "z_proj" in h2:
            process_llm.z_proj.load_state_dict(h2["z_proj"])
        if "protocol_head" in h2:
            process_llm.protocol_head.load_state_dict(h2["protocol_head"])
        if "bridge" in h2:
            bridge.load_state_dict(h2["bridge"])
    process_llm.z_proj.to(dev, dtype=torch.bfloat16)
    process_llm.protocol_head.to(dev, dtype=torch.bfloat16)
    bridge.to(dev, dtype=torch.bfloat16)
    process_llm.freeze_for_stage3()
    for p in bridge.parameters():
        p.requires_grad_(False)
    bridge.eval()

    # 5. Compute z_proto per paper (cache)
    log.info("computing z_proto for %d papers ...", len(paper_methods_path))
    z_proto_per_paper: dict[str, torch.Tensor] = {}
    for i, (pid, mpath) in enumerate(sorted(paper_methods_path.items())):
        if pid not in paper_to_keys:
            continue  # paper not referenced in V58 CSV
        try:
            text = mpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        with torch.no_grad():
            z = process_llm.extract_z_proto([text])
        z_proto_per_paper[pid] = z.squeeze(0).detach().to("cpu", dtype=torch.float32)
        if (i + 1) % 50 == 0:
            log.info("  z_proto %d/%d", i + 1, len(paper_methods_path))

    # 6. For each V58 key, build the physics prompt + run PhysicsLLM once,
    #    then compute z_LLM = Bridge(z_proto, z_phys) for each (paper, key)
    #    pair and average over papers sharing this key.
    from src.data.stage1_qa_dataset import build_prompt as build_physics_prompt

    log.info("computing z_phys + z_LLM for %d keys ...",
             len(needed_per_key_papers))
    keys_out: list[str] = []
    z_llm_out: list[torch.Tensor] = []
    for j, (key, pids) in enumerate(sorted(needed_per_key_papers.items())):
        cation, atm, T_bin = key.split("|")
        # Build the physics prompt for (cation, atm_bucket, T_bin)
        physics_inp = {
            "dopant": cation,
            "dopant_concentration": 0.026,   # canonical 2.6% — V58 cache is keyed
            "atmosphere": atm,
            "temperature_C": float(T_bin),
            "fermi_level_eV": 4.0,
        }
        prompt = build_physics_prompt(physics_inp)
        with torch.no_grad():
            z_phys = physics_llm.extract_z_phys([prompt]).squeeze(0).to("cpu", dtype=torch.float32)

        # Average over papers
        z_llm_sum = torch.zeros(32, dtype=torch.float32)
        n_papers = 0
        for pid in pids:
            if pid not in z_proto_per_paper:
                continue
            z_proto = z_proto_per_paper[pid]
            with torch.no_grad():
                z_llm = bridge(
                    z_proto.unsqueeze(0).to(dev, dtype=torch.bfloat16),
                    z_phys.unsqueeze(0).to(dev, dtype=torch.bfloat16),
                ).squeeze(0).to("cpu", dtype=torch.float32)
            z_llm_sum += z_llm
            n_papers += 1
        if n_papers == 0:
            continue
        keys_out.append(key)
        z_llm_out.append(z_llm_sum / n_papers)
        if (j + 1) % 25 == 0:
            log.info("  keys %d/%d", j + 1, len(needed_per_key_papers))

    feats = torch.stack(z_llm_out, dim=0).numpy().astype("float32")  # [K, 32]
    keys_arr = np.array(keys_out, dtype="<U32")

    meta = {
        "key_scheme": "{dopant}|{atm_bucket}|{T_bin_C}",
        "atm_buckets": list(_V58_ATM_O2FRAC.keys()),
        "T_bins_C": _V58_T_BINS,
        "n_keys": int(len(keys_out)),
        "n_papers_processed": int(len(z_proto_per_paper)),
        "v58_csv": args.v58_csv,
        "stage1_adapter": args.stage1_adapter,
        "stage2_adapter": args.stage2_adapter,
        "z_dim": 32,
        "paper_id_in_key": False,
        "notes": (
            "Per (dopant, atm_bucket, T_bin) key, z_LLM is averaged over the "
            "set of papers whose v58 CSV row matched that key. The averaging "
            "implicitly absorbs cross-paper protocol variance into the cached "
            "vector — exactly what V58 FiLM consumes via _lookup_z_llm."
        ),
    }

    np.savez(
        out_path,
        keys=keys_arr,
        features=feats,  # P0 fix (Audit 6): Ga2O3Net._load_dft_cache reads dl["features"]
        meta_json=np.array(json.dumps(meta), dtype=object),
    )
    log.info("wrote %s (K=%d, dim=32)", out_path, len(keys_out))
    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "n_keys": len(keys_out),
        "n_papers_processed": len(z_proto_per_paper),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
