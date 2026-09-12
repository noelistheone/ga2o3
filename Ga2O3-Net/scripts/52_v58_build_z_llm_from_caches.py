"""
Phase-58 LLMBridge refinement — Step 3/3: build the z_LLM cache.

Computes  z_LLM = trained_bridge(z_proto, z_phys)  for every
"{dopant}|{atm}|{T_C}" key that V58 Stage-3 actually needs (= the set of
unique non-NONE `dft_features_idx` values in the v58 CSV — exactly what
`Ga2O3Net._v58_dopant_atm_T_key` produces and `_lookup_z_llm` looks up),
averaging z_LLM over the papers that contribute that key (papers whose
dopant element == the key's cation, via the v56 markdown manifest source_pdf
path tag).

Neither LLM is loaded here: z_proto and z_phys both come from the on-disk
caches (scripts 50 and 43), the only learnable piece is the 25k bridge
(script 51). Pure-CPU/GPU-fast.

Output:  data/processed/z_llm_cache_v58.npz
  keys     : [K]   "<U32" array of "{dopant}|{atm}|{T}" keys
  features : [K,32] float32   (NOT 'feats' — Audit-6 P0 fix: Ga2O3Net.
             _load_dft_cache reads dl["features"])
  meta_json: JSON blob

Keys whose cation has NO contributing methods-paper, or whose z_phys is absent
from the z_phys cache, are omitted — `_lookup_z_llm` zero-fills missing keys,
which is the documented V58-lite-DFT fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _norm_el(el: str) -> str:
    return el[0].upper() + el[1:].lower() if el else el


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--z-proto-cache", type=str,
                   default="data/processed/z_proto_cache_v58.npz")
    p.add_argument("--z-phys-cache", type=str,
                   default="data/processed/z_phys_cache_v58.npz")
    p.add_argument("--bridge", type=str,
                   default="checkpoints/qwen25_v58_bridge/bridge.pt")
    p.add_argument("--v58-csv", type=str,
                   default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv")
    p.add_argument("--md-manifest", type=str,
                   default="extra-paper/v56_markdown/manifest.csv")
    p.add_argument("--out", type=str,
                   default="data/processed/z_llm_cache_v58.npz")
    p.add_argument("--gpu", type=int, default=1, help="GPU 1 ONLY (task). -1=CPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import pandas as pd
    import torch

    from src.models.llm_bridge import LLMBridge

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("52_z_llm")

    dev = torch.device("cuda:0" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. caches ─────────────────────────────────────────────────────────────
    zp_proto = np.load(HERE / args.z_proto_cache, allow_pickle=True)
    proto_keys = [str(k) for k in zp_proto["keys"]]
    proto_feats = zp_proto["features"].astype(np.float32)
    proto_map = {k: proto_feats[i] for i, k in enumerate(proto_keys)}
    log.info("z_proto cache: %d papers", len(proto_map))

    zp_phys = np.load(HERE / args.z_phys_cache, allow_pickle=True)
    phys_map = {k: np.asarray(zp_phys[k], dtype=np.float32)
                for k in zp_phys.files if "|" in k}
    log.info("z_phys cache: %d keys", len(phys_map))

    # ── 2. paper -> dopant element ────────────────────────────────────────────
    mdman = pd.read_csv(HERE / args.md_manifest)
    dopant_re = re.compile(r"/doping/([A-Za-z]{1,3})/")
    papers_by_cation: dict[str, list[str]] = defaultdict(list)
    for _, r in mdman.iterrows():
        m = dopant_re.search(str(r["source_pdf"]))
        if not m:
            continue
        pid = str(r["doi_hash"]).strip()
        if pid in proto_map:
            papers_by_cation[_norm_el(m.group(1))].append(pid)
    log.info("cations with contributing papers: %d", len(papers_by_cation))

    # ── 3. downstream-needed keys ─────────────────────────────────────────────
    df = pd.read_csv(HERE / args.v58_csv)
    csv_keys = df["dft_features_idx"].astype(str)
    needed_keys = sorted(set(csv_keys[(csv_keys != "NONE")
                                      & (csv_keys.str.contains(r"\|"))]))
    log.info("downstream-needed keys: %d", len(needed_keys))

    # ── 4. bridge ─────────────────────────────────────────────────────────────
    ck = torch.load(HERE / args.bridge, map_location="cpu")
    bridge = LLMBridge(dim=ck.get("dim", 32), n_heads=ck.get("n_heads", 4),
                       n_layers=ck.get("n_layers", 2), dropout=0.0)
    bridge.load_state_dict(ck["bridge"])
    bridge.to(dev).eval()
    log.info("loaded trained bridge from %s", args.bridge)

    # ── 5. per key: average z_LLM over contributing papers ────────────────────
    keys_out, feats_out = [], []
    skipped_no_phys, skipped_no_paper = [], []
    for key in needed_keys:
        cation = key.split("|")[0]
        if key not in phys_map:
            skipped_no_phys.append(key)
            continue
        pids = papers_by_cation.get(cation, [])
        if not pids:
            skipped_no_paper.append(key)
            continue
        z_phys = torch.tensor(phys_map[key], device=dev).unsqueeze(0)   # [1,32]
        acc = torch.zeros(32, device=dev)
        n = 0
        for pid in pids:
            z_proto = torch.tensor(proto_map[pid], device=dev).unsqueeze(0)
            with torch.no_grad():
                z_llm = bridge(z_proto, z_phys).squeeze(0)
            acc += z_llm
            n += 1
        feats_out.append((acc / n).to("cpu", dtype=torch.float32).numpy())
        keys_out.append(key)

    if not keys_out:
        raise RuntimeError("no z_LLM keys produced")

    feats = np.stack(feats_out, 0).astype(np.float32)   # [K, 32]
    keys_arr = np.array(keys_out, dtype="<U32")
    assert np.isfinite(feats).all(), "z_LLM features contain NaN/Inf"

    meta = {
        "key_scheme": "{dopant}|{atm_bucket}|{T_bin_C}",
        "n_keys": int(len(keys_out)),
        "n_needed_keys": int(len(needed_keys)),
        "skipped_no_zphys": skipped_no_phys,
        "skipped_no_contributing_paper": skipped_no_paper,
        "z_dim": 32,
        "paper_id_in_key": False,
        "bridge": str(args.bridge),
        "z_proto_cache": str(args.z_proto_cache),
        "z_phys_cache": str(args.z_phys_cache),
        "notes": (
            "z_LLM = trained_bridge(z_proto, z_phys), averaged over papers whose "
            "dopant element matches the key cation. Missing keys are zero-filled "
            "by Ga2O3Net._lookup_z_llm (V58-lite-DFT fallback)."
        ),
    }
    np.savez(out_path, keys=keys_arr, features=feats,
             meta_json=np.array(json.dumps(meta), dtype=object))
    log.info("wrote %s  (K=%d, dim=32)", out_path, len(keys_out))

    # ── 6. verify load contract ───────────────────────────────────────────────
    chk = np.load(out_path, allow_pickle=False)
    assert "features" in chk.files and "keys" in chk.files
    assert chk["features"].shape == (len(keys_out), 32)
    assert chk["keys"].shape == (len(keys_out),)
    assert np.isfinite(chk["features"]).all()

    print(json.dumps({
        "ok": True,
        "out": str(out_path),
        "n_keys": len(keys_out),
        "n_needed": len(needed_keys),
        "skipped_no_zphys": skipped_no_phys,
        "skipped_no_contributing_paper": skipped_no_paper,
        "features_shape": list(chk["features"].shape),
        "keys_shape": list(chk["keys"].shape),
        "load_verified": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
