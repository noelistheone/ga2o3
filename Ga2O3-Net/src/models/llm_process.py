"""
M5: Process-LLM (ProcessLLM).

Wraps ``Qwen/Qwen2.5-7B-Instruct`` (hidden size 3584) with a 4-bit (NF4)
QLoRA adapter and two trainable heads:

  1. a 3584 → 32 projection producing the prototype/process embedding
     ``z_proto`` (masked mean-pooled last hidden state, then a single
     bias-free Linear); and
  2. a multi-label protocol classification head (§5.2.3) emitting 12 binary
     protocol logits.

Reference: design doc §3.5 (esp. §3.5.2 LoRA config, §3.5.4 embedding
extraction, §3.5.5 output) and §5.2.3 (12 binary protocol labels), plus
§1.4 H1/H2/H3 constraints.

Training schedule (§5.2): LoRA + projection + multi-label head + the
cross-attention bridge are trained in **Stage 2**, then FROZEN for the
downstream V_O/PDR regression (Stage 3) per the H2 parameter-budget
constraint. See ``freeze_for_stage3()``.

The 12 multi-label protocol classes (§5.2.3) cover, e.g.: target purity
bin, Ar:O2 ratio bin, contact metal type, anneal atmosphere, illumination
wavelength bin, bias voltage bin (the doc names six families spanning 12
binary labels total).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Hyperparameters (design doc §3.5) ─────────────────────────────────────────
PROCESS_LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROCESS_HIDDEN_SIZE = 3584       # Qwen2.5-7B hidden size (§3.5.4)
Z_PROTO_DIM = 32                 # z_proto embedding dim (§3.5.5)
MAX_SEQ_LEN = 4096               # §3.5.1
N_PROTOCOL_LABELS = 12           # §5.2.3 — 12 binary protocol labels

# LoRA config (§3.5.2)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class _TinyToyTransformer(nn.Module):
    """
    2-layer toy causal transformer used ONLY by the CPU self-test
    (``tiny_test=True``). Mimics the parts of a HF causal LM that
    ``ProcessLLM`` touches: an ``embed_tokens`` layer and a forward returning
    ``output.hidden_states[-1]`` of shape [B, T, hidden]. Avoids loading the
    real 7B weights for a shape/wiring test.
    """

    def __init__(self, hidden: int, vocab: int = 256, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=2 * hidden,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        h = self.embed_tokens(input_ids)
        key_padding = None
        if attention_mask is not None:
            key_padding = attention_mask == 0  # True at pad positions
        h = self.layers(h, src_key_padding_mask=key_padding)

        class _Out:
            pass

        out = _Out()
        out.hidden_states = (h,)
        return out


class ProcessLLM(nn.Module):
    """
    Process-LLM (M5): QLoRA-wrapped Qwen2.5-7B + z_proto projection +
    12-way multi-label protocol head.

    Build either:
      - ``ProcessLLM(tiny_test=True)`` — toy backbone, CPU, no downloads; or
      - ``ProcessLLM.from_pretrained_qlora(...)`` — real 4-bit QLoRA model.

    Args:
        backbone   : the language model (HF PEFT model, or toy transformer).
        tokenizer  : HF tokenizer (None in tiny_test mode).
        hidden_size: backbone hidden size (3584 for Qwen2.5-7B).
        max_len    : tokenizer truncation length (§3.5.1, default 4096).
        tiny_test  : if True, build a 2-layer toy transformer instead.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        tokenizer=None,
        hidden_size: int = PROCESS_HIDDEN_SIZE,
        max_len: int = MAX_SEQ_LEN,
        tiny_test: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.tiny_test = tiny_test
        self.tokenizer = tokenizer

        if tiny_test:
            self.backbone = backbone or _TinyToyTransformer(hidden=hidden_size)
        else:
            if backbone is None:
                raise ValueError(
                    "ProcessLLM requires a backbone unless tiny_test=True. "
                    "Use ProcessLLM.from_pretrained_qlora(...) for the real model."
                )
            self.backbone = backbone

        # z_proto projection: 3584 -> 32, bias-free (§3.5.4). Trained in Stage 2,
        # frozen in Stage 3.
        self.z_proj = nn.Linear(hidden_size, Z_PROTO_DIM, bias=False)

        # multi-label protocol classification head (§5.2.3): 12 binary logits.
        # Trained in Stage 2 (BCE-with-logits), frozen in Stage 3.
        self.protocol_head = nn.Linear(hidden_size, N_PROTOCOL_LABELS)

    # ── Embedding extraction (§3.5.4) ─────────────────────────────────────────

    def _pooled_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean-pool of the last hidden state over non-pad tokens.

        Exact form from §3.5.4:
            last_hidden = outputs.hidden_states[-1]       # [B, seq, 3584]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]            # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(1).clamp_min(1.0)
        pooled = (last_hidden * mask).sum(1) / denom       # [B, H]
        return pooled

    def extract_z_proto(self, methods_texts: list[str]) -> torch.Tensor:
        """
        Tokenize ``methods_texts`` (max_len 4096), forward, masked mean-pool
        the last hidden state (§3.5.4), then project 3584 → 32.

        Args:
            methods_texts: list of paper Methods-section prompts (length B).

        Returns:
            z_proto: [B, 32].
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "extract_z_proto needs a tokenizer; build via from_pretrained_qlora."
            )
        enc = self.tokenizer(
            methods_texts, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding=True,
        )
        device = next(self.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pooled = self._pooled_hidden(input_ids, attention_mask)
        return self.z_proj(pooled)

    def protocol_logits(self, methods_texts: list[str]) -> torch.Tensor:
        """Tokenize → pooled → 12 binary protocol logits (§5.2.3)."""
        if self.tokenizer is None:
            raise RuntimeError("protocol_logits needs a tokenizer.")
        enc = self.tokenizer(
            methods_texts, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding=True,
        )
        device = next(self.parameters()).device
        pooled = self._pooled_hidden(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        return self.protocol_head(pooled)

    def forward_from_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Token-level entry point (used by the self-test and Stage-2 training,
        which pre-tokenizes). Returns ``(z_proto, protocol_logits)``.
        """
        pooled = self._pooled_hidden(input_ids, attention_mask)
        z_proto = self.z_proj(pooled)
        logits = self.protocol_head(pooled)
        return z_proto, logits

    # ── Stage-3 freezing (H2 parameter-budget constraint, §1.4) ───────────────

    def n_trainable(self) -> int:
        """Number of trainable (requires_grad=True) parameters across the whole module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_trainable_breakdown(self) -> dict[str, int]:
        """Trainable param count split into LoRA / projection / protocol head."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        lora = sum(
            p.numel() for n, p in self.backbone.named_parameters()
            if p.requires_grad and "lora_" in n.lower()
        )
        return {
            "lora": lora,
            "z_proj": _count(self.z_proj),
            "protocol_head": _count(self.protocol_head),
            "total": self.n_trainable(),
        }

    def freeze_for_stage3(self) -> None:
        """
        Freeze ALL params (LoRA, projection, protocol head) and switch to eval.

        After this call ``n_trainable()`` must be 0 — M5 contributes no
        trainable params to the downstream V_O/PDR regression (§1.4 H2).
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ── Real model builder (guarded so import works without a GPU) ────────────

    @classmethod
    def from_pretrained_qlora(
        cls,
        model_name: str = PROCESS_LLM_NAME,
        device_map: str | dict | None = "auto",
        max_len: int = MAX_SEQ_LEN,
        gradient_checkpointing: bool = True,
        init_adapter_dir: str | None = None,
    ) -> "ProcessLLM":
        """
        Build the real 4-bit (NF4) QLoRA Process-LLM.

        Imports of bitsandbytes / transformers / peft happen INSIDE this method
        so the module imports cleanly on a GPU-less machine.

        Args:
            model_name           : HF model id (default Qwen2.5-7B-Instruct).
            device_map           : passed to ``from_pretrained`` (e.g. "auto",
                                   ``{"": "cuda:0"}``).
            max_len              : tokenizer truncation length (§3.5.1).
            gradient_checkpointing: enable for Stage-2 memory savings.

        Returns:
            A ``ProcessLLM`` wrapping the PEFT-wrapped 4-bit backbone.
        """
        import torch as _torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            PeftModel,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=_torch.bfloat16,
        )
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=gradient_checkpointing
        )

        if init_adapter_dir is not None:
            # Resume: load existing LoRA weights (crash-resilience / continue training).
            peft_model = PeftModel.from_pretrained(
                base, init_adapter_dir, is_trainable=True
            )
        else:
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            peft_model = get_peft_model(base, lora_config)

        hidden_size = getattr(peft_model.config, "hidden_size", PROCESS_HIDDEN_SIZE)
        return cls(
            backbone=peft_model,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            max_len=max_len,
            tiny_test=False,
        )


def _estimate_lora_params(
    hidden: int, n_layers: int, r: int,
    intermediate: int, kv_dim: int, target_modules: list[str],
) -> int:
    """
    Analytic LoRA trainable-param estimate for the real Qwen2.5-7B
    (used only to print an expectation; the toy backbone has no LoRA).
    For a Linear(in,out), LoRA adds r*(in+out) params.
    Qwen2.5-7B: hidden=3584, n_layers=28, intermediate=18944,
    GQA -> k_proj/v_proj out = 512 (4 KV heads * 128 head_dim).
    """
    per_layer = 0
    if "q_proj" in target_modules:
        per_layer += r * (hidden + hidden)
    if "k_proj" in target_modules:
        per_layer += r * (hidden + kv_dim)
    if "v_proj" in target_modules:
        per_layer += r * (hidden + kv_dim)
    if "o_proj" in target_modules:
        per_layer += r * (hidden + hidden)
    if "gate_proj" in target_modules:
        per_layer += r * (hidden + intermediate)
    if "up_proj" in target_modules:
        per_layer += r * (hidden + intermediate)
    if "down_proj" in target_modules:
        per_layer += r * (intermediate + hidden)
    return per_layer * n_layers


def _self_test() -> None:
    """CPU self-test with a TINY toy backbone (no 7B load, no GPU)."""
    torch.manual_seed(0)

    tiny_hidden = 64
    model = ProcessLLM(tiny_test=True, hidden_size=tiny_hidden)

    print("[ProcessLLM self-test]")
    proj = sum(p.numel() for p in model.z_proj.parameters())
    head = sum(p.numel() for p in model.protocol_head.parameters())
    print(f"  z_proj params (tiny hidden={tiny_hidden})        : {proj:,}")
    print(f"  protocol_head params (tiny hidden={tiny_hidden}) : {head:,}")

    # Report REAL-config head/projection sizes (hidden=3584) for the report.
    real_proj = PROCESS_HIDDEN_SIZE * Z_PROTO_DIM  # bias-free
    real_head = (PROCESS_HIDDEN_SIZE + 1) * N_PROTOCOL_LABELS
    print(f"  [real cfg] z_proj (3584->32, no bias)      : {real_proj:,}")
    print(f"  [real cfg] protocol_head (3584->12)        : {real_head:,}")
    lora_est = _estimate_lora_params(
        hidden=PROCESS_HIDDEN_SIZE, n_layers=28, r=LORA_R,
        intermediate=18944, kv_dim=512, target_modules=LORA_TARGET_MODULES,
    )
    print(f"  [real cfg] LoRA est (r=16, 28 layers)      : {lora_est:,} (~{lora_est/1e6:.2f} M)")

    # forward via token ids on the toy backbone
    B, T = 4, 16
    input_ids = torch.randint(0, 256, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[1, T // 2:] = 0  # exercise padding mask

    z_proto, logits = model.forward_from_ids(input_ids, attention_mask)
    assert z_proto.shape == (B, Z_PROTO_DIM), f"z_proto shape {z_proto.shape}"
    assert logits.shape == (B, N_PROTOCOL_LABELS), f"logits shape {logits.shape}"
    assert torch.isfinite(z_proto).all() and torch.isfinite(logits).all()
    print(f"  extract z_proto          : {tuple(z_proto.shape)} (OK)")
    print(f"  protocol logits          : {tuple(logits.shape)} (OK)")

    # gradient flows to projection + protocol head
    loss = z_proto.pow(2).sum() + logits.pow(2).sum()
    loss.backward()
    assert model.z_proj.weight.grad is not None, "no grad to z_proj"
    assert model.protocol_head.weight.grad is not None, "no grad to protocol_head"
    print(f"  trainable (toy)          : {model.n_trainable():,}")

    # freeze_for_stage3 drops trainable to 0
    model.freeze_for_stage3()
    assert model.n_trainable() == 0, "freeze_for_stage3 left trainable params"
    assert not model.training
    print(f"  after freeze             : trainable={model.n_trainable()} eval={not model.training}")
    print("[ProcessLLM self-test] PASS")


if __name__ == "__main__":
    _self_test()
