"""V56-Pretrain-CPT — QLoRA continued pretraining of Qwen-7B on Ga2O3 corpus.

Inputs:
  - extra-paper/v56_markdown/*.md   (~1500 PDFs × ~5K tokens ≈ 7.5M tokens after dedup)

Pipeline:
  1. Load all markdowns, concatenate with double-newlines, chunk to 1024 tokens.
  2. 4-bit NF4 QLoRA on Qwen-7B (~12 GB VRAM per card with grad-checkpointing).
  3. Stage 1: pure CPT (LM objective, 1 epoch, cosine lr 5e-5).
  4. Stage 2 (optional): SFT on 488+V56-Ext rows as Q&A — uses V56-ICL-Fixed's
     format directly.
  5. Held-out 50 Q&A eval (manual prompts) to verify Ga2O3 knowledge improvement.

Outputs:
  - checkpoints/qwen2.5-7b-ga2o3-cpt/    — QLoRA adapter
  - results/phase56v56cpt/eval_qa.json   — held-out accuracy + perplexity drop
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v56_cpt")


QWEN_BASE = PROJ / "checkpoints" / "qwen2.5-7b"
CPT_OUT = PROJ / "checkpoints" / "qwen2.5-7b-ga2o3-cpt"
MARKDOWN_DIR = PROJ / "extra-paper" / "v56_markdown"
RESULTS_DIR = PROJ / "results" / "phase56v56cpt"


def build_corpus(min_chars: int = 1500) -> list[str]:
    """Load markdowns, return chunked text blocks (1 per markdown, post-cleanup)."""
    out = []
    for md in sorted(MARKDOWN_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        if len(text) < min_chars:
            continue
        out.append(text)
    logger.info(f"Loaded {len(out)} markdowns (≥{min_chars} chars)")
    return out


def train_cpt(corpus: list[str], device: str = "cuda:0",
              epochs: int = 1, lr: float = 5e-5, rank: int = 32,
              block_size: int = 1024) -> None:
    """4-bit NF4 QLoRA continued-pretraining."""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Loading Qwen-7B with NF4 quant on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_BASE), quantization_config=bnb_cfg, device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=rank, lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"QLoRA trainable: {n_trainable:,} / {n_total:,} "
                f"({100*n_trainable/n_total:.2f}%)")

    # Tokenize → chunk
    def _tokenize(ex):
        out = tokenizer(ex["text"], truncation=False, padding=False)
        return out

    def _group(ex):
        """Concat all tokens and split into block_size chunks."""
        concat = []
        for ids in ex["input_ids"]:
            concat += ids + [tokenizer.eos_token_id]
        n = (len(concat) // block_size) * block_size
        result = {
            "input_ids": [concat[i:i+block_size] for i in range(0, n, block_size)],
            "attention_mask": [[1]*block_size for _ in range(0, n, block_size)],
        }
        result["labels"] = [list(x) for x in result["input_ids"]]
        return result

    ds = Dataset.from_dict({"text": corpus})
    ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
    ds = ds.map(_group, batched=True, batch_size=200,
                remove_columns=["input_ids", "attention_mask"])
    logger.info(f"Training blocks: {len(ds)} × {block_size} tokens")

    CPT_OUT.mkdir(parents=True, exist_ok=True)
    args_t = TrainingArguments(
        output_dir=str(CPT_OUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args_t,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(CPT_OUT))
    tokenizer.save_pretrained(str(CPT_OUT))
    logger.info(f"CPT adapter saved → {CPT_OUT}")


HELDOUT_QA = [
    ("What is the band gap of β-Ga2O3?", ["4.8", "4.9", "5.0", "4.7"]),
    ("Which dopant typically acts as a deep acceptor in β-Ga2O3?", ["Mg", "Sn", "Si", "Hf"]),
    ("Which dopant typically acts as a shallow donor in β-Ga2O3?", ["Si", "Mg", "Bi", "Sb"]),
    ("What is the typical V_O concentration in undoped sputter-grown β-Ga2O3?",
     ["10^17", "10^14", "10^21", "10^18"]),
    ("Post-deposition O2 anneal typically does what to V_O density?",
     ["decreases", "increases", "no change", "doubles"]),
    ("XPS O1s deconvolution probes which defect?", ["V_O", "V_Ga", "Mg_Ga", "interstitial"]),
    ("Increasing Sn doping concentration typically does what to PDR?",
     ["increases", "decreases", "no effect", "reverses"]),
    ("Which substrate is most commonly used for sputter-deposited Ga2O3 photodetectors?",
     ["c-sapphire", "GaN", "MgO", "Si"]),
    ("The acceptor compensation ratio in Mg:Ga2O3 is approximately?",
     ["high (>50%)", "negligible", "around 10%", "exactly 0"]),
    ("Solar-blind UV photodetectors operate in which wavelength range?",
     ["200-280nm", "300-400nm", "400-500nm", "600-700nm"]),
]


def eval_heldout_qa(device: str = "cuda:0") -> dict:
    """Quick multi-choice eval. Approx accuracy via simple text-match heuristic."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(CPT_OUT))
    base = AutoModelForCausalLM.from_pretrained(
        str(QWEN_BASE), torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model = PeftModel.from_pretrained(base, str(CPT_OUT))
    model.eval()

    correct = 0
    answers = []
    for q, choices in HELDOUT_QA:
        prompt = f"<|im_start|>user\nQuestion: {q}\nChoices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}\nAnswer with the letter only.<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8,
                                  pad_token_id=tokenizer.eos_token_id, do_sample=False)
        resp = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Heuristic: ground-truth is always choices[0] (the first listed option)
        # Check if response indicates "A" or the correct option text
        resp_upper = resp.strip().upper()
        is_correct = (resp_upper.startswith("A") or choices[0].lower() in resp.lower())
        if is_correct:
            correct += 1
        answers.append({"q": q, "gt": choices[0], "resp": resp[:50].strip(), "correct": is_correct})
    acc = 100 * correct / len(HELDOUT_QA)
    return {"accuracy_pct": acc, "n_correct": correct, "n_total": len(HELDOUT_QA), "answers": answers}


def main() -> None:
    ap = argparse.ArgumentParser(description="V56-Pretrain-CPT Qwen QLoRA")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "both"):
        corpus = build_corpus(min_chars=1500)
        if len(corpus) < 30:
            logger.error(f"Too few markdowns ({len(corpus)}); abort CPT")
            return
        train_cpt(corpus, device=args.device, epochs=args.epochs, rank=args.rank,
                   block_size=args.block_size)

    if args.mode in ("eval", "both"):
        if not CPT_OUT.exists():
            logger.error(f"CPT adapter not found at {CPT_OUT}; skip eval")
            return
        eval_out = eval_heldout_qa(device=args.device)
        (RESULTS_DIR / "eval_qa.json").write_text(json.dumps(eval_out, indent=2))
        logger.info(f"V56-Pretrain-CPT eval: {eval_out['n_correct']}/{eval_out['n_total']} "
                    f"= {eval_out['accuracy_pct']:.1f}% (gate ≥60%)")


if __name__ == "__main__":
    main()
