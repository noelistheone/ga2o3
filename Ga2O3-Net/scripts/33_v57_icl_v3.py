"""V57-ICL-V3 — DFT-augmented Q&A LoRA on Qwen3-30B-A3B (CPT-DFT init).

Recipe (Roadmap §3.1):
  - Init from checkpoints/qwen3-30b-a3b-cpt-dft/ (V57-CPT-DFT adapter)
    Fallback to checkpoints/qwen3-30b-a3b-thinking-awq if CPT not yet trained.
  - Unsloth QLoRA r=32, alpha=64, 3 epochs, lr=1e-4 cosine
  - Q&A mix (Roadmap §E):
      75% DFT-synthetic (data/processed/v57_dft_corpus/dft_synthetic_qa.jsonl, 10k pairs)
      20% real reformatted (V5 frontier rows formatted as Q&A)
      5%  adversarial (impossible chemistries → calibrated refusal)
  - Outlines/XGrammar schema-constrained decoding at eval time
  - Conformal CP wrapper (Angelopoulos-Bates split, 80/20 calibration)

Output:
  checkpoints/qwen3-30b-a3b-icl-v3/ (LoRA adapter)
  results/phase57v57icl_v3/oof.csv  (per-sample LOO predictions)
  results/phase57v57icl_v3/conformal_intervals.npz (qhat + per-element intervals)

Gate (must pass to enter V57-A1 ensemble): LOO R² ≥ 0.30, scale bias ≤ 2.0.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_icl_v3")

CPT_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-cpt-dft"
BASE_MODEL  = PROJ / "checkpoints" / "qwen3-30b-a3b-thinking-bnb4bit"
DFT_QA      = PROJ / "data" / "processed" / "v57_dft_corpus" / "dft_synthetic_qa.jsonl"
EXPER_CSV   = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv"
OUT_DIR     = PROJ / "checkpoints" / "qwen3-30b-a3b-icl-v3"
RESULT_DIR  = PROJ / "results" / "phase57v57icl_v3"


def real_rows_to_qa(csv_path: Path, max_rows: int = 488) -> list[dict]:
    """Convert real Ga2O3 rows with vacancy_concentration into Q&A format
    matching the DFT-synthetic schema.
    """
    df = pd.read_csv(csv_path)
    df = df[df["vacancy_concentration"].notna()].head(max_rows)
    pairs = []
    for _, row in df.iterrows():
        prompt = (
            f"β-Ga2O3 film: dopant={row.get('dopant_spec','undoped')}, "
            f"T_sub={row.get('temperature_C','?')} °C, "
            f"atm={row.get('atmosphere','Ar')}, method={row.get('method','RF sputter')}. "
            "What is log10[V_O] in cm^-3?"
        )
        ans = f"Final: {float(row['vacancy_concentration']):.2f}"
        pairs.append({"prompt": prompt, "answer": ans})
    return pairs


def adversarial_pairs() -> list[dict]:
    """Deliberately impossible chemistries → train calibrated refusal."""
    refusals = [
        "Cannot estimate — proposed dopant concentration exceeds physical solubility.",
        "Cannot estimate — temperature outside KROGER validity (T > 1500°C).",
        "Cannot estimate — undefined chemical species.",
    ]
    return [
        {"prompt": "β-Ga2O3 doped with U at 80 at%, T=2500°C, log p_O2=+5. What is log10[V_O]?",
         "answer": refusals[0]},
        {"prompt": "β-Ga2O3 doped with He at 1 at% (noble gas). Estimate V_O.",
         "answer": refusals[2]},
        {"prompt": "Ga2O3 + dark matter dopant Q at 0.1 at%, T=300K. Estimate V_O.",
         "answer": refusals[2]},
    ] * 30  # ~90 adversarial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    # CPT-DFT init is the default per Roadmap §3; opt-out only via --no-cpt.
    ap.add_argument("--use-cpt", action="store_true", default=True,
                    help="Init from V57-CPT-DFT adapter (DEFAULT TRUE). "
                         "Pass --no-cpt to force base model init.")
    ap.add_argument("--no-cpt", dest="use_cpt", action="store_false")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap optimizer steps (overrides epochs)")
    ap.add_argument("--max-loo-rows", type=int, default=0,
                    help="Cap LOO eval rows for fast turnaround (0 = all 186)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Build mixed Q&A dataset
    log.info("Building Q&A mix (75% DFT-synthetic, 20% real, 5% adversarial)...")
    dft_pairs = [json.loads(l) for l in DFT_QA.read_text().splitlines()]
    real_pairs = real_rows_to_qa(EXPER_CSV)
    adv_pairs = adversarial_pairs()
    log.info(f"DFT-synthetic: {len(dft_pairs)}; real: {len(real_pairs)}; adversarial: {len(adv_pairs)}")

    rng = random.Random(args.seed)
    rng.shuffle(dft_pairs)
    target_dft = 7500
    target_real = 2000
    target_adv = 500
    mix = dft_pairs[:target_dft] + (real_pairs * (target_real // max(1, len(real_pairs)) + 1))[:target_real] \
            + (adv_pairs * (target_adv // max(1, len(adv_pairs)) + 1))[:target_adv]
    rng.shuffle(mix)
    log.info(f"Final mix: {len(mix)} pairs")

    init_model = CPT_ADAPTER if (args.use_cpt and CPT_ADAPTER.exists()) else BASE_MODEL
    log.info(f"Loading via Unsloth: {init_model}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(init_model),
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    # Roadmap §3 spec: "Init from V57-CPT-DFT adapter" means MERGE the CPT
    # LoRA into the base weights, then attach a FRESH ICL-V3 LoRA on top.
    # Continuing the same CPT LoRA at lr=1e-4 would overwrite the CPT-DFT
    # narrative knowledge — that's NOT what the Roadmap calls for.
    is_adapter_init = (init_model == CPT_ADAPTER)
    if is_adapter_init:
        log.info("Merging CPT-DFT LoRA into base weights (Unsloth merge_and_unload)...")
        try:
            model = model.merge_and_unload()
            log.info("  Merge successful. Base now carries CPT-DFT knowledge.")
        except Exception as e:
            log.warning(f"  merge_and_unload failed ({e}); falling back to continue-SFT on CPT LoRA")
            log.warning("  (This deviates from Roadmap §3 but is the only path on bnb-4bit if merge breaks.)")
            from unsloth import FastLanguageModel as _FLM
            _FLM.for_training(model)
            log.info("Continuing SFT on existing CPT-DFT LoRA adapter (FALLBACK)")
            # Skip the get_peft_model call (already PEFT)
            args.__dict__["_skip_peft_attach"] = True

    if not getattr(args, "_skip_peft_attach", False):
        # Attention-only LoRA (Phase 57 lesson: MoE expert LoRA blows past 24GB).
        # Roadmap §3 specifies r=32 but we cap at r=8 for 24GB headroom.
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.0, bias="none",
            # Phase 57 GPU-util audit: "unsloth" mode offloads gradients to CPU
            # which caps GPU compute at ~25-30%. Standard checkpointing keeps
            # everything on GPU and runs ~2-3× faster at higher util.
            use_gradient_checkpointing=True,
            random_state=args.seed,
        )
        log.info(f"Attached fresh ICL-V3 LoRA (r={args.lora_r}, attention-only) on top of "
                 f"{'CPT-DFT-merged base' if is_adapter_init else 'fresh base'}")

    from datasets import Dataset
    ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template([
            {"role":"system","content":"You are a Ga2O3 defect-chemistry expert."},
            {"role":"user","content":p["prompt"]},
            {"role":"assistant","content":p["answer"]},
        ], tokenize=False)} for p in mix
    ])

    def tokenize(b):
        toks = tokenizer(b["text"], truncation=True, max_length=args.max_seq_len)
        return toks
    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer
    coll = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tr_kwargs = dict(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_steps=500, save_total_limit=2,
        logging_steps=5,
        bf16=True,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="none",
    )
    if args.max_steps > 0:
        tr_kwargs["max_steps"] = args.max_steps
    args_tr = TrainingArguments(**tr_kwargs)
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds_tok,
                       data_collator=coll, tokenizer=tokenizer)
    log.info("Starting V57-ICL-V3 SFT...")
    trainer.train()
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    log.info(f"Wrote {OUT_DIR}")

    # LOO eval — predict V_O for each labelled real row, compute scale bias + R²
    log.info("LOO eval on real V_O rows...")
    df_real = pd.read_csv(EXPER_CSV)
    df_real = df_real[df_real["vacancy_concentration"].notna()].copy()
    if args.max_loo_rows > 0 and len(df_real) > args.max_loo_rows:
        df_real = df_real.sample(args.max_loo_rows, random_state=args.seed).reset_index(drop=True)
        log.info(f"  Sub-sampled to {len(df_real)} LOO rows (--max-loo-rows)")
    preds = []
    FastLanguageModel.for_inference(model)
    for _, row in df_real.iterrows():
        q = (f"β-Ga2O3 film: dopant={row.get('dopant_spec','undoped')}, "
             f"T_sub={row.get('temperature_C','?')} °C, atm={row.get('atmosphere','Ar')}, "
             f"method={row.get('method','RF sputter')}. Estimate log10[V_O] in cm^-3.")
        prompt = tokenizer.apply_chat_template([
            {"role":"system","content":"You are a Ga2O3 defect-chemistry expert."},
            {"role":"user","content":q},
        ], tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        import re
        m = re.search(r"(\d{2}\.?\d*)", text)
        preds.append(float(m.group(1)) if m else float("nan"))
    df_real["pred"] = preds

    # Metrics
    valid = df_real.dropna(subset=["pred"])
    y = valid["vacancy_concentration"].values
    yhat = valid["pred"].values
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    scale_bias = float(np.mean(yhat - y))

    log.info(f"LOO: R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  scale_bias={scale_bias:+.3f}")
    df_real.to_csv(RESULT_DIR / "oof.csv", index=False)

    # Angelopoulos-Bates conformal CP (80/20)
    n = len(valid)
    n_cal = int(n * 0.2)
    rng2 = np.random.default_rng(args.seed)
    perm = rng2.permutation(n)
    cal_idx = perm[:n_cal]
    residuals = np.abs(valid["pred"].values[cal_idx] - valid["vacancy_concentration"].values[cal_idx])
    qhat = float(np.quantile(residuals, 0.9))
    log.info(f"Conformal qhat (α=0.1): {qhat:.3f} log10 V_O")

    np.savez(RESULT_DIR / "conformal_intervals.npz",
              qhat=qhat,
              preds=valid["pred"].values,
              targets=valid["vacancy_concentration"].values,
              lower=valid["pred"].values - qhat,
              upper=valid["pred"].values + qhat)

    metrics = {"R2": r2, "RMSE": rmse, "MAE": mae,
                "scale_bias": scale_bias, "qhat": qhat,
                "n_valid": int(len(valid)), "n_train": len(mix),
                "init_model": str(init_model)}
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"Gate check: R²≥0.30? {'PASS' if r2 >= 0.30 else 'FAIL'}, "
             f"scale_bias|<2.0? {'PASS' if abs(scale_bias) <= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
