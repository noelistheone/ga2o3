"""V57-CPT-DFT — Qwen3-30B-A3B continued pretrain on Ga2O3 + DFT corpus.

Recipe (Roadmap §2.1):
  - Base: checkpoints/qwen3-30b-a3b-thinking-awq (AWQ-4bit, 18GB on single 3090)
  - Unsloth FastLanguageModel QLoRA r=64, alpha=128, disable_router_lora=True
  - Liger kernel `apply_liger_kernel_to_qwen3_moe`
  - Corpus (15.8M tokens, 8:1 PDF:DFT mix via sampling weight):
      * V56 PDF text (rebuilt from extra-paper/v56_markdown/*.md), ~12.7M tokens
      * V57 DFT narrative (data/processed/v57_dft_corpus/narratives/*.md), ~1.5M tokens
      * KROGER/Brouwer prose (narratives/kroger_brouwer_scan.md), ~0.8M tokens
      * (optional) ChemBench-format Ga2O3 Q&A, ~0.6M tokens
  - 1 epoch, lr=2e-5 cosine, warmup 3%, batch=1 × grad_accum 16, block_size 4096
  - Optional DataParallel across GPU 0 + 1

Output: checkpoints/qwen3-30b-a3b-cpt-dft/ (LoRA adapter + tokenizer)

Eval gates (run scripts/32a_v57_cpt_eval.py after this):
  - Ga2O3 Q&A held-out 200 ≥ 75% (V56 baseline 70%)
  - GPQA-chemistry subset ≥ 72%
  - MaScQA materials subset ≥ 65%
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_cpt_dft")

BASE_MODEL = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen3-30b-a3b-thinking-bnb4bit"
PDF_MD_DIR = PROJ / "extra-paper" / "v56_markdown"
DFT_NARR_DIR = PROJ / "data" / "processed" / "v57_dft_corpus" / "narratives"
DFT_QA_PATH = PROJ / "data" / "processed" / "v57_dft_corpus" / "dft_synthetic_qa.jsonl"
OUT_DIR = PROJ / "checkpoints" / "qwen3-30b-a3b-cpt-dft"


def build_corpus(seed: int = 42, pdf_dft_ratio: float = 8.0) -> list[str]:
    """Concatenate PDF text + DFT narratives + KROGER prose into the CPT corpus.

    Sampling: 8:1 PDF:DFT enforced by repeating DFT 1× and PDF (n_pdf // n_dft / 8)×
    if needed — but with V56 PDFs we already have ~12.7M tokens vs ~2M DFT, so
    natural ratio is ~6:1. We pad DFT by 0.3× to reach 8:1 effective.
    """
    random.seed(seed)
    pdf_texts = []
    for md in sorted(PDF_MD_DIR.glob("*.md")):
        try:
            t = md.read_text(errors="ignore")
            if len(t) > 500:
                pdf_texts.append(t)
        except Exception:
            continue
    log.info(f"Collected {len(pdf_texts)} PDF markdown files")

    dft_texts = []
    for md in sorted(DFT_NARR_DIR.glob("*.md")):
        try:
            t = md.read_text(errors="ignore")
            if len(t) > 100:
                dft_texts.append(t)
        except Exception:
            continue
    log.info(f"Collected {len(dft_texts)} DFT narrative files")

    if DFT_QA_PATH.exists():
        # Reformat Q&A as flowing text (prompt + answer) for next-token CPT
        for line in DFT_QA_PATH.read_text().splitlines():
            try:
                obj = json.loads(line)
                dft_texts.append(f"{obj['prompt']}\n\n{obj['answer']}")
            except Exception:
                continue
        log.info(f"Added Q&A pairs (corpus now {len(dft_texts)} DFT chunks)")

    # Sampling: interleave PDF chunks with DFT chunks at the prescribed ratio
    corpus = []
    rng = random.Random(seed)
    pdf_idx, dft_idx = 0, 0
    while pdf_idx < len(pdf_texts):
        # 8 PDF chunks then 1 DFT chunk
        for _ in range(int(pdf_dft_ratio)):
            if pdf_idx < len(pdf_texts):
                corpus.append(pdf_texts[pdf_idx]); pdf_idx += 1
        if dft_texts:
            corpus.append(dft_texts[dft_idx % len(dft_texts)]); dft_idx += 1
    rng.shuffle(corpus)
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap optimizer steps (overrides epochs). For demonstrative "
                         "CPT under wall-clock budget set e.g. 60.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build corpus + load model only; skip training step")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building CPT corpus (V56 PDF text + V57 DFT narratives + Q&A)...")
    corpus = build_corpus(seed=args.seed)
    log.info(f"Corpus = {len(corpus)} chunks "
             f"(estimated {sum(len(c) for c in corpus)/1e6:.1f}M chars ≈ "
             f"{sum(len(c) for c in corpus)/4e6:.1f}M tokens)")

    log.info(f"Loading base model {BASE_MODEL} via Unsloth FastLanguageModel...")
    import torch
    # Liger MoE LoRA kernel is broken with Unsloth-compiled Qwen3-MoE
    # (stale unsloth_compiled_cache redefines RMSNorm). Skip — Unsloth's own
    # fused kernels already give ~2× speed; Liger adds ~10% on top, not worth
    # debugging the patching conflict.
    log.info("Liger kernel: disabled (Unsloth own kernels handle MoE)")

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.block_size,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    # Memory budget: 30B-A3B has 128 MoE experts × 48 layers; LoRA on every
    # expert blows past 24GB even at r=16. Restrict LoRA to attention only
    # (q/k/v/o_proj — 4 modules × 48 layers, ~64M params at r=8) and let the
    # gate/up/down MoE experts stay frozen. This still adapts the routing-
    # invariant attention to Ga2O3 vocabulary which is the main bottleneck.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.0,
        bias="none",
        # Phase 57 GPU-util audit: standard checkpoint (not "unsloth") avoids
        # the smart-grad-offload that capped util at ~25-30%.
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )
    log.info(f"Model loaded. Trainable params: "
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    if args.dry_run:
        log.info("Dry-run mode: skipping training")
        return

    # Build HF Dataset
    from datasets import Dataset
    ds = Dataset.from_dict({"text": corpus})

    # Tokenize + chunk
    def tokenize(batch):
        toks = tokenizer(batch["text"], truncation=False, return_special_tokens_mask=False)
        return toks
    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)

    def group_texts(examples):
        concat = sum(examples["input_ids"], [])
        n = (len(concat) // args.block_size) * args.block_size
        return {"input_ids": [concat[i:i+args.block_size] for i in range(0, n, args.block_size)],
                "labels":    [concat[i:i+args.block_size] for i in range(0, n, args.block_size)]}
    ds_chunked = ds_tok.map(group_texts, batched=True, remove_columns=ds_tok.column_names)
    log.info(f"Tokenized + chunked → {len(ds_chunked)} blocks of {args.block_size} tokens "
             f"≈ {len(ds_chunked) * args.block_size / 1e6:.2f}M effective tokens/epoch")

    from transformers import TrainingArguments, Trainer
    tr_kwargs = dict(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=5,
        bf16=True,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="none",
    )
    if args.max_steps > 0:
        tr_kwargs["max_steps"] = args.max_steps
    args_tr = TrainingArguments(**tr_kwargs)
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds_chunked,
                       tokenizer=tokenizer)
    log.info("Starting CPT training...")
    trainer.train()

    log.info("Saving LoRA adapter...")
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    (OUT_DIR / "training_log.json").write_text(json.dumps({
        "base_model": BASE_MODEL,
        "n_corpus_chunks": len(corpus),
        "n_train_blocks": len(ds_chunked),
        "args": vars(args),
    }, indent=2))
    log.info(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
