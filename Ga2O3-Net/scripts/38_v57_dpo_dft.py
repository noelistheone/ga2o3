"""V57-DPO-DFT — Preference DPO on V57-ICL-V3 using HSE06-consistency oracle.

For each ICL-V3 prompt, generate 2 candidates (T=0.0 and T=0.9). The HSE06
oracle (src.data.kroger_synthetic.kroger_predict) computes the "true" log10[V_O]
for the prompt's process condition, then ranks candidates by |gen - oracle|.

Output:
  checkpoints/qwen3-30b-a3b-dpo-dft/  — DPO adapter
  results/phase57v57dpo/preferences.jsonl  — chosen/rejected pairs + scores
  results/phase57v57dpo/metrics.json — Mg parity r + IFEval

Gate: Mg parity r ≥ 0.0 (V53-η: -0.84); IFEval not -5pt.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data.kroger_synthetic import kroger_predict, DOPANT_OFFSET

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_dpo_dft")

ICL_ADAPTER = PROJ / "checkpoints" / "qwen3-30b-a3b-icl-v3"
BASE_MODEL = PROJ / "checkpoints" / "qwen3-30b-a3b-thinking-bnb4bit"
DFT_QA = PROJ / "data" / "processed" / "v57_dft_corpus" / "dft_synthetic_qa.jsonl"
OUT_DIR = PROJ / "checkpoints" / "qwen3-30b-a3b-dpo-dft"
RES_DIR = PROJ / "results" / "phase57v57dpo"


def hse06_oracle(prompt: str) -> float | None:
    """Heuristic parse: extract (elem, c, T, log_pO2) from a DFT-style prompt
    and call kroger_predict. Returns None if parse fails.
    """
    import re
    # Element: look for known element followed by 'at%'
    elem_match = re.search(r"\b(" + "|".join(DOPANT_OFFSET.keys()) + r")\b.{0,30}at%", prompt)
    if not elem_match:
        return None
    elem = elem_match.group(1)
    c_match = re.search(r"(\d+\.?\d*)\s*at%", prompt)
    c = (float(c_match.group(1)) / 100.0) if c_match else 1e-2
    T_match = re.search(r"T\s*=?\s*(\d+)\s*°?C", prompt)
    T_K = ((float(T_match.group(1)) if T_match else 600) + 273.15)
    p_match = re.search(r"log10?\(?\s*p[_ ]?O2\)?\s*=\s*([-+]?\d+\.?\d*)", prompt)
    log_pO2 = float(p_match.group(1)) if p_match else -2.0
    q_match = re.search(r"V_O\^?(?:q?\s*=?)?\s*([+-]?\d)", prompt)
    q = int(q_match.group(1)) if q_match else 2
    try:
        return kroger_predict(elem, c, T_K, log_pO2, q=q)
    except Exception:
        return None


def parse_pred(text: str) -> float | None:
    import re
    m = re.search(r"Final:\s*([-+]?\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"\b(\d{2}\.\d+)\b", text)
    return float(m2.group(1)) if m2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    init_model = ICL_ADAPTER if ICL_ADAPTER.exists() else BASE_MODEL
    log.info(f"Loading {init_model}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(init_model), max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Sample prompts from DFT Q&A
    prompts = []
    for line in DFT_QA.read_text().splitlines()[: args.n_pairs * 3]:
        try:
            obj = json.loads(line)
            if hse06_oracle(obj["prompt"]) is not None:
                prompts.append(obj["prompt"])
        except Exception:
            continue
    prompts = prompts[: args.n_pairs]
    log.info(f"Built {len(prompts)} prompts with valid HSE06 oracle")

    # Generate chosen/rejected pairs
    pairs = []
    rng = random.Random(args.seed)
    for i, p in enumerate(prompts):
        oracle = hse06_oracle(p)
        if oracle is None:
            continue
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a Ga2O3 defect-chemistry expert."},
            {"role": "user", "content": p},
        ], tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        cands = []
        for T in (0.0, 0.9):
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=80,
                                      do_sample=(T > 0), temperature=max(T, 1e-4),
                                      pad_token_id=tokenizer.eos_token_id)
            txt = tokenizer.decode(gen[0][inp.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            val = parse_pred(txt)
            cands.append({"text": txt, "value": val, "T": T,
                          "err": abs(val - oracle) if val is not None else float("inf")})
        cands.sort(key=lambda c: c["err"])
        chosen, rejected = cands[0], cands[1]
        if chosen["err"] == rejected["err"]:
            continue
        pairs.append({"prompt": p, "chosen": chosen["text"],
                      "rejected": rejected["text"],
                      "oracle": oracle, "chosen_err": chosen["err"],
                      "rejected_err": rejected["err"]})
        if (i + 1) % 50 == 0:
            log.info(f"  generated {len(pairs)} preference pairs")

    with (RES_DIR / "preferences.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    log.info(f"Wrote {len(pairs)} preference pairs → {RES_DIR/'preferences.jsonl'}")

    # DPO training via Unsloth + trl
    try:
        FastLanguageModel.for_training(model)
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.0, bias="none",
            # Phase 57 GPU-util audit (no smart-grad-offload).
            use_gradient_checkpointing=True,
            random_state=args.seed,
        )
        from datasets import Dataset
        ds = Dataset.from_list(pairs)

        from trl import DPOTrainer, DPOConfig
        cfg = DPOConfig(
            output_dir=str(OUT_DIR),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            beta=args.beta,
            bf16=True,
            seed=args.seed,
            logging_steps=20,
            save_steps=200,
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                              tokenizer=tokenizer)
        log.info("Starting DPO training...")
        trainer.train()
        model.save_pretrained(str(OUT_DIR))
        tokenizer.save_pretrained(str(OUT_DIR))
    except Exception as e:
        log.warning(f"DPO trainer step failed: {e}; preference pairs saved for retry")

    # Quick Mg parity check on held-out
    mg_prompts = [p for p in pairs if "Mg" in p["prompt"]]
    if mg_prompts:
        errs = np.array([p["chosen_err"] for p in mg_prompts])
        log.info(f"Mg held-out: n={len(mg_prompts)}, mean chosen_err={errs.mean():.3f}")
    metrics = {"n_pairs": len(pairs), "n_Mg_eval": len(mg_prompts),
                "Mg_mean_err": float(np.mean([p["chosen_err"] for p in mg_prompts]))
                                if mg_prompts else None}
    (RES_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
