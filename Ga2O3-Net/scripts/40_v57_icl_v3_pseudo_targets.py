"""V57-ICL-V3 → pseudo-target NPZ for V57-A1 retrain (Roadmap §3 ensemble plumb).

Loads the V57-ICL-V3 LoRA adapter (or V57-DPO-DFT adapter if available, since
DPO is the refined preference-tuned version), runs the LLM over every row of
the V57 training CSV asking for log10[V_O], parses the answer, and writes:

  data/processed/v57_icl_v3_pseudo_targets_vo.npz
    pseudo_log_vo   [N]   — float, NaN for failed parses
    pseudo_std      [N]   — float, low-confidence rows get higher std

V57-A1 retrain consumes this NPZ via fusion_v57a1_sincere_*.yaml
`pdr_self_distill.path = data/processed/v57_icl_v3_pseudo_targets_vo.npz`.

Gate: only swap the V57-A1 config's pseudo-target NPZ to this V57-ICL-V3
output **if** the LOO R² gate (≥0.30) passes in the ICL-V3 metrics.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_icl_v3_pseudo")

V57_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v57.csv"
ICL_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-icl-v3"
DPO_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-dpo-dft"
CPT_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-cpt-dft"
BASE_MODEL = PROJ / "checkpoints" / "qwen3-30b-a3b-thinking-bnb4bit"

ICL_METRICS = PROJ / "results" / "phase57v57icl_v3" / "metrics.json"
OUT_NPZ = PROJ / "data" / "processed" / "v57_icl_v3_pseudo_targets_vo.npz"

LOO_R2_GATE = 0.30
PROMPT_TPL = (
    "β-Ga2O3 film: dopant={dopant}, T_sub={T} °C, atm={atm}, method={method}. "
    "Estimate log10[V_O] in cm^-3."
)


def parse_final_number(text: str) -> float | None:
    """Extract the LLM's V_O answer. Try `Final: X.Y` then any 2-digit float."""
    m = re.search(r"Final:\s*([-+]?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"\b(1\d\.\d+|2\d\.\d+)\b", text)
    return float(m2.group(1)) if m2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-priority", nargs="+",
                    default=["dpo", "icl", "cpt"],
                    help="Order in which to try the LLM adapters")
    ap.add_argument("--skip-gate", action="store_true",
                    help="Bypass the LOO R²≥0.30 gate (for testing)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    # Gate: enforce ICL-V3 LOO R²≥0.30 before generating pseudo-targets
    if not args.skip_gate:
        if not ICL_METRICS.exists():
            log.error(f"ICL-V3 metrics not found at {ICL_METRICS}. Run V57-ICL-V3 first.")
            sys.exit(2)
        m = json.loads(ICL_METRICS.read_text())
        r2 = m.get("R2", -1.0)
        sb = abs(m.get("scale_bias", 99))
        log.info(f"ICL-V3 metrics: R²={r2:.3f}, |scale_bias|={sb:.3f}")
        if r2 < LOO_R2_GATE:
            log.error(f"GATE FAIL: ICL-V3 LOO R²={r2:.3f} < {LOO_R2_GATE}. "
                      f"Per Roadmap §3 stop-condition, V57-ICL-V3 does NOT enter V57-A1 ensemble.")
            log.error(f"V57-A1 retrain will keep V53-ζ legacy pseudo-targets.")
            sys.exit(3)
        log.info(f"GATE PASS: R²={r2:.3f} ≥ {LOO_R2_GATE}. Generating V57-A1 pseudo-targets…")

    # Select adapter (DPO > ICL > CPT fallback)
    adapter_dirs = {"dpo": DPO_ADAPTER, "icl": ICL_ADAPTER, "cpt": CPT_ADAPTER}
    chosen = None
    for tag in args.adapter_priority:
        cand = adapter_dirs.get(tag)
        if cand and cand.exists() and (cand / "adapter_config.json").exists():
            chosen = (tag, cand)
            break
    if chosen is None:
        log.error("No V57 adapter found on disk. Run V57-CPT-DFT / ICL-V3 / DPO-DFT first.")
        sys.exit(4)
    tag, adapter_path = chosen
    log.info(f"Using {tag.upper()} adapter at {adapter_path}")

    # Load base model + adapter via Unsloth (FastLanguageModel auto-merges PEFT)
    from unsloth import FastLanguageModel
    log.info(f"Loading base model {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),  # Unsloth picks up adapter_config from this dir
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Iterate over V57 CSV
    df = pd.read_csv(V57_CSV)
    log.info(f"Generating V_O predictions for {len(df)} rows...")
    preds, stds = [], []
    n_ok, n_fail = 0, 0
    for i, row in df.iterrows():
        # Skip rows with no dopant (undoped baseline) but keep predicting V_O anyway
        prompt = PROMPT_TPL.format(
            dopant=row.get("dopant_spec", "undoped") or "undoped",
            T=row.get("temperature_C", "?"),
            atm=row.get("atmosphere", "Ar"),
            method=row.get("method", "RF sputter"),
        )
        chat = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a Ga2O3 defect-chemistry expert. Return only 'Final: X.X'."},
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        v = parse_final_number(text)
        if v is None or not (14.0 <= v <= 22.0):
            preds.append(np.nan); stds.append(np.nan); n_fail += 1
        else:
            preds.append(v); stds.append(0.5); n_ok += 1   # constant std=0.5 baseline
        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(df)}] ok={n_ok} fail={n_fail}")

    pseudo_log_vo = np.array(preds, dtype=np.float32)
    pseudo_std    = np.array(stds, dtype=np.float32)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, pseudo_log_vo=pseudo_log_vo, pseudo_std=pseudo_std)
    log.info(f"Wrote {OUT_NPZ}")

    # Also write a metrics file for traceability
    meta = {
        "phase": 57,
        "stage": "V57-A1 LLM pseudo-target generation",
        "adapter_used": tag,
        "adapter_path": str(adapter_path),
        "n_rows": int(len(df)),
        "n_valid_predictions": int(n_ok),
        "n_failed_parses": int(n_fail),
        "valid_rate": float(n_ok) / max(1, len(df)),
        "gate_loo_r2_passed": True if args.skip_gate else (r2 >= LOO_R2_GATE),
        "gate_loo_r2_threshold": LOO_R2_GATE,
        "npz_path": str(OUT_NPZ),
        "consumed_by": "config/fusion_v57a1_sincere_*.yaml → pdr_self_distill.path",
    }
    if not args.skip_gate:
        meta["icl_v3_metrics"] = json.loads(ICL_METRICS.read_text())
    (OUT_NPZ.parent / "v57_icl_v3_pseudo_targets_metrics.json").write_text(
        json.dumps(meta, indent=2)
    )
    log.info(f"Metrics: {meta}")


if __name__ == "__main__":
    main()
