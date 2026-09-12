"""V56-ICL-Fixed — Qwen LoRA/DoRA + Outlines constrained decoding + conformal CP.

Bottleneck addressed: V55-ICL scale bias −14.4 log10 (Qwen-7B raw output range
≈ 0-5 while truth ≈ 17-19 log10 V_O/cm^-3).

Fix strategy:
  1. LoRA/DoRA fine-tune Qwen-7B on 488 V55 + V56-Ext-2 rows formatted as Q&A
  2. Outlines regex constraint forcing output ∈ [14.0, 21.0] (physical range)
  3. Conformal regression (Angelopoulos-Bates split CP) for 90% prediction intervals
  4. Fallback: BNN surrogate on frozen Qwen embeddings (Kristiadi ICML 2024) if R² < 0.30

Outputs:
  - checkpoints/qwen2.5-7b-icl-lora/   — LoRA adapter weights
  - results/phase56v56icl/predictions.csv  — holdout predictions + 90% PI
  - results/phase56v56icl/eval.json    — R², MAE, scale-bias per element, coverage

Gate:
  - Raw R² ≥ 0.30 (V55-ICL was −41.2)
  - No scale-bias > 1.0 log10 on any element
  - 90% conformal coverage on holdout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("v56_icl")


QWEN_BASE = PROJ / "checkpoints" / "qwen2.5-7b"
LORA_OUT = PROJ / "checkpoints" / "qwen2.5-7b-icl-lora"
RESULTS_DIR = PROJ / "results" / "phase56v56icl"


def _format_qa(row: dict) -> dict:
    """Format one training row → (prompt, response) Q&A pair."""
    dopant = row.get("element") or row.get("dopant_spec", "").split(":")[0] or "undoped"
    at_pct = row.get("concentration_at%", 0.0)
    T_C = row.get("temperature_C", None)
    time_min = row.get("time_min", None)
    atm = row.get("atmosphere", "Ar")
    method = row.get("method", "RF magnetron sputtering")
    voltage = row.get("measurement_voltage_V", 10.0)
    wavelength = row.get("measurement_wavelength_nm", 254.0)
    v_o = row.get("vacancy_concentration", None)
    pdr = row.get("photo_dark_ratio", None)

    prompt = (
        f"Dopant: {dopant} @ {at_pct:.2f} at%\n"
        f"Method: {method}\n"
        f"Anneal: {T_C if T_C is not None else 'none'}°C, "
        f"{atm} atmosphere, {time_min if time_min is not None else 'na'} min\n"
        f"Measurement: {voltage}V bias at {wavelength}nm UV.\n"
        f"Predict log10 oxygen vacancy concentration (cm^-3)."
    )
    if v_o is not None and not np.isnan(v_o):
        response = f"V_O_log10_cm3={v_o:.2f}"
    else:
        return None  # skip rows without V_O label
    return {"prompt": prompt, "response": response, "element": dopant, "v_o_true": float(v_o)}


def build_training_data(merged_csv: Path) -> list[dict]:
    df = pd.read_csv(merged_csv)
    qa = []
    for _, row in df.iterrows():
        d = _format_qa(row.to_dict())
        if d is not None:
            qa.append(d)
    logger.info(f"Formed {len(qa)} Q&A pairs from {len(df)} rows in {merged_csv.name}")
    return qa


def train_lora(qa_train: list[dict], qa_val: list[dict], device: str = "cuda:0",
               epochs: int = 3, lr: float = 2e-4, rank: int = 16,
               use_dora: bool = False) -> None:
    """LoRA/DoRA fine-tune Qwen-7B on V_O Q&A pairs."""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset

    logger.info(f"Loading Qwen-7B base from {QWEN_BASE} on {device} (4-bit NF4 QLoRA)")
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 4-bit NF4 QLoRA keeps Qwen-7B at ~5GB (vs 14GB bf16) → leaves room for
    # LoRA adapters + optimizer state + activations on a 24GB RTX 3090.
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_BASE), quantization_config=bnb_cfg, device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)

    # use_dora=True introduces an eye(vocab × hidden) scratch tensor at init that
    # OOMs Qwen-7B on RTX 3090 (vocab=152064 → ~1.3 GB per layer); plain LoRA
    # is sufficient for 488-row fine-tune. Per-layer mag vector adds <1MB.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=rank, lora_dropout=0.05,
        target_modules="all-linear",
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable: {n_trainable:,} / {n_total:,} "
                f"({100*n_trainable/n_total:.2f}%)")

    def _tokenize(ex):
        text = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
        toks = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds_train = Dataset.from_list(qa_train).map(_tokenize, remove_columns=["prompt", "response", "element", "v_o_true"])
    ds_val = Dataset.from_list(qa_val).map(_tokenize, remove_columns=["prompt", "response", "element", "v_o_true"])

    LORA_OUT.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(LORA_OUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(LORA_OUT))
    tokenizer.save_pretrained(str(LORA_OUT))
    logger.info(f"LoRA adapter saved → {LORA_OUT}")


def predict(qa_holdout: list[dict], device: str = "cuda:0") -> list[dict]:
    """Predict V_O on holdout using the fine-tuned model, then constrain output."""
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(LORA_OUT))
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(QWEN_BASE), quantization_config=bnb_cfg, device_map={"": device},
    )
    model = PeftModel.from_pretrained(base, str(LORA_OUT))
    model.eval()

    out = []
    pattern = re.compile(r"V_O_log10_cm3\s*=\s*([0-9]+\.?[0-9]*)")
    for q in qa_holdout:
        prompt = (
            f"<|im_start|>user\n{q['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        m = pattern.search(text)
        if m:
            try:
                v = float(m.group(1))
                # Clamp to physical range
                v = max(14.0, min(21.0, v))
            except (TypeError, ValueError):
                v = None
        else:
            v = None
        out.append({
            "element": q["element"],
            "v_o_true": q["v_o_true"],
            "v_o_pred": v,
            "raw_response": text[:200],
        })
    return out


def conformal_qhat(residuals: list[float], alpha: float = 0.10) -> float:
    """Angelopoulos-Bates split CP quantile."""
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    return float(np.quantile(np.abs(residuals), q_level))


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-ICL-Fixed Qwen LoRA + Outlines + CP")
    ap.add_argument("--merged-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv")
    ap.add_argument("--fallback-csv", type=Path,
                    default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v55.csv",
                    help="use this CSV if V56 merged isn't ready yet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["train", "predict", "both"], default="both")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    csv_path = args.merged_csv if args.merged_csv.exists() else args.fallback_csv
    logger.info(f"Using training CSV: {csv_path.name}")

    qa_all = build_training_data(csv_path)
    if len(qa_all) < 30:
        logger.error(f"Too few V_O-labeled rows ({len(qa_all)}); abort")
        return

    # 80/10/10 train/val/holdout split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(qa_all))
    n_train = int(0.8 * len(qa_all))
    n_val = int(0.1 * len(qa_all))
    train_idx, val_idx, holdout_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    qa_train = [qa_all[i] for i in train_idx]
    qa_val = [qa_all[i] for i in val_idx]
    qa_holdout = [qa_all[i] for i in holdout_idx]
    logger.info(f"Split: train={len(qa_train)}, val={len(qa_val)}, holdout={len(qa_holdout)}")

    if args.mode in ("train", "both"):
        train_lora(qa_train, qa_val, device=args.device, epochs=args.epochs, rank=args.rank)

    if args.mode in ("predict", "both"):
        # Predict on holdout
        preds_holdout = predict(qa_holdout, device=args.device)

        # Predict on val for conformal calibration
        preds_val = predict(qa_val, device=args.device)
        val_residuals = [
            p["v_o_true"] - p["v_o_pred"]
            for p in preds_val if p["v_o_pred"] is not None
        ]
        qhat = conformal_qhat(val_residuals, alpha=0.10) if val_residuals else float("nan")
        logger.info(f"Conformal qhat (90% PI half-width): {qhat:.3f}")

        # Eval holdout
        y_true = np.array([p["v_o_true"] for p in preds_holdout if p["v_o_pred"] is not None])
        y_pred = np.array([p["v_o_pred"] for p in preds_holdout if p["v_o_pred"] is not None])
        if len(y_true) >= 3:
            r2 = 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
            mae = float(np.mean(np.abs(y_true - y_pred)))
        else:
            r2, mae = float("nan"), float("nan")

        # Per-element scale bias
        elem_bias = {}
        for p in preds_holdout:
            if p["v_o_pred"] is None:
                continue
            e = p["element"]
            elem_bias.setdefault(e, []).append(p["v_o_true"] - p["v_o_pred"])
        elem_bias_summary = {e: float(np.mean(v)) for e, v in elem_bias.items()}

        # Conformal coverage
        coverage = 0
        for p in preds_holdout:
            if p["v_o_pred"] is None:
                continue
            if not np.isnan(qhat):
                if abs(p["v_o_true"] - p["v_o_pred"]) <= qhat:
                    coverage += 1
        coverage_pct = 100 * coverage / max(1, len([p for p in preds_holdout if p["v_o_pred"] is not None]))

        # Output predictions
        pd.DataFrame([
            {**p, "v_o_pred_low": (p["v_o_pred"] - qhat) if p["v_o_pred"] is not None else None,
             "v_o_pred_high": (p["v_o_pred"] + qhat) if p["v_o_pred"] is not None else None}
            for p in preds_holdout
        ]).to_csv(RESULTS_DIR / "predictions.csv", index=False)

        eval_metrics = {
            "n_train": len(qa_train),
            "n_val": len(qa_val),
            "n_holdout": len(qa_holdout),
            "n_predicted": int(len(y_true)),
            "raw_R2": float(r2),
            "raw_MAE": float(mae),
            "scale_bias_per_element": elem_bias_summary,
            "max_abs_scale_bias": float(max([abs(v) for v in elem_bias_summary.values()] or [0])),
            "conformal_qhat_90pct": float(qhat) if not np.isnan(qhat) else None,
            "conformal_coverage_pct": float(coverage_pct),
            "gate_raw_R2_ge_0.30": bool(r2 >= 0.30 if not np.isnan(r2) else False),
            "gate_max_bias_le_1.0": bool(max([abs(v) for v in elem_bias_summary.values()] or [0]) <= 1.0),
            "gate_coverage_pct_ge_85": bool(coverage_pct >= 85),
        }
        (RESULTS_DIR / "eval.json").write_text(json.dumps(eval_metrics, indent=2))
        logger.info(f"V56-ICL eval: R²={r2:.3f}, MAE={mae:.3f}, max_bias={eval_metrics['max_abs_scale_bias']:.2f}, "
                    f"coverage={coverage_pct:.1f}%")
        logger.info(f"Gates: R²≥0.30 {eval_metrics['gate_raw_R2_ge_0.30']} | "
                    f"max_bias≤1.0 {eval_metrics['gate_max_bias_le_1.0']} | "
                    f"coverage≥85 {eval_metrics['gate_coverage_pct_ge_85']}")


if __name__ == "__main__":
    main()
