"""V57-ICL-V3 fast LOO eval — separate from training so we don't waste
training compute on slow generation. Loads the saved ICL-V3 adapter,
generates short answers (max_new_tokens=16), and computes LOO R² + scale
bias + conformal qhat. Writes results/phase57v57icl_v3/metrics.json which
gates scripts/40.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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
log = logging.getLogger("v57_icl_loo")

ICL_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-icl-v3"
EXPER_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v57.csv"
RESULT_DIR = PROJ / "results" / "phase57v57icl_v3"


def parse_pred(text: str) -> float | None:
    m = re.search(r"Final:\s*([-+]?\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"\b(1[4-9]\.\d+|2[0-2]\.\d+)\b", text)
    return float(m2.group(1)) if m2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-loo-rows", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXPER_CSV)
    df = df[df["vacancy_concentration"].notna()].copy()
    if args.max_loo_rows > 0 and len(df) > args.max_loo_rows:
        df = df.sample(args.max_loo_rows, random_state=args.seed).reset_index(drop=True)
    log.info(f"LOO eval on {len(df)} V_O rows")

    from unsloth import FastLanguageModel
    log.info(f"Loading ICL-V3 adapter from {ICL_ADAPTER}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ICL_ADAPTER),
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    preds = []
    t0 = pd.Timestamp.now()
    for i, row in df.iterrows():
        q = (f"β-Ga2O3 film: dopant={row.get('dopant_spec','undoped')}, "
             f"T_sub={row.get('temperature_C','?')} °C, atm={row.get('atmosphere','Ar')}, "
             f"method={row.get('method','RF sputter')}. Estimate log10[V_O] in cm^-3.")
        chat = tokenizer.apply_chat_template([
            {"role":"system","content":"You are a Ga2O3 defect-chemistry expert. Return only 'Final: X.X'."},
            {"role":"user","content":q},
        ], tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inp.input_ids.shape[1]:],
                                skip_special_tokens=True)
        v = parse_pred(text)
        preds.append(v if v is not None else float("nan"))
        if (i + 1) % 5 == 0:
            elapsed = (pd.Timestamp.now() - t0).total_seconds()
            log.info(f"  [{i+1}/{len(df)}] {elapsed/(i+1):.1f}s/row avg")

    df["pred"] = preds
    valid = df.dropna(subset=["pred"])
    y = valid["vacancy_concentration"].values
    yhat = valid["pred"].values
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    scale_bias = float(np.mean(yhat - y))

    df.to_csv(RESULT_DIR / "oof.csv", index=False)

    # Conformal qhat (split 80/20)
    rng2 = np.random.default_rng(args.seed)
    n = len(valid)
    n_cal = max(2, int(n * 0.2))
    perm = rng2.permutation(n)
    cal = perm[:n_cal]
    residuals = np.abs(yhat[cal] - y[cal])
    qhat = float(np.quantile(residuals, 0.9)) if len(residuals) > 0 else float("nan")

    np.savez(RESULT_DIR / "conformal_intervals.npz",
             qhat=qhat, preds=yhat, targets=y,
             lower=yhat - qhat, upper=yhat + qhat)

    m = {
        "phase": 57,
        "stage": "V57-ICL-V3 (LoRA r=8 attention-only on CPT-merged Qwen3-30B-Thinking)",
        "backend": "Qwen3-30B-A3B-Thinking-2507 BNB-4bit + ICL-V3 LoRA",
        "n_loo_rows": int(len(df)),
        "n_valid_predictions": int(len(valid)),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "scale_bias": scale_bias,
        "qhat_alpha_0.1": qhat,
        "gate_loo_r2_ge_030_AND_scale_bias_abs_le_2.0":
            (r2 >= 0.30) and (abs(scale_bias) <= 2.0),
        "gate_threshold_r2": 0.30,
        "gate_threshold_scale_bias": 2.0,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_sec": (pd.Timestamp.now() - t0).total_seconds(),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(m, indent=2))
    log.info(f"=== ICL-V3 LOO ===")
    log.info(f"  R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, scale_bias={scale_bias:+.3f}")
    log.info(f"  Valid predictions: {len(valid)}/{len(df)}")
    log.info(f"  Gate (R²≥0.30 AND |scale_bias|≤2.0): "
             f"{'PASS' if m['gate_loo_r2_ge_030_AND_scale_bias_abs_le_2.0'] else 'FAIL'}")


if __name__ == "__main__":
    main()
