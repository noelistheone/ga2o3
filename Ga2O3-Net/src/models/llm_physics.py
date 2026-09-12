"""
M6: Physics-LLM (PhysicsLLM).

Wraps ``Qwen/Qwen2.5-1.5B-Instruct`` (hidden size 1536) with a 4-bit
(NF4) QLoRA adapter and two trainable heads:

  1. a 1536 → 32 projection producing the physics embedding ``z_phys``
     (mean-pooled last hidden state, then a single bias-free Linear); and
  2. a numeric prior head (§3.6.4) that regresses defect-chemistry numerics.

Reference: design doc §3.6 (esp. §3.6.2 LoRA config, §3.6.4 numeric prior
head, §3.6.5 output) and §1.4 H1/H2/H3 constraints.

Training schedule (§3.6.4 / §5.1): LoRA + projection + numeric head are
trained in **Stage 1**, then FROZEN for the downstream V_O/PDR regression
(Stage 3) per the H2 parameter-budget constraint. See ``freeze_for_stage3()``.

Numeric prior head — output split note
---------------------------------------
§3.6.4 lists FOUR *semantic* numeric outputs:
    - predicted log_vo               (scalar)
    - predicted dominant_charge      (3-way softmax over V_O charge 0/+1/+2)
    - predicted log_K_eq             (scalar)
    - predicted defect_formation_energy (scalar)
The doc sketches the head as "[1536 → 256 → 4]". That width is only
APPROXIMATE: because ``dominant_charge`` is a 3-way classification it needs
3 logits, so the 4 semantic outputs actually require 1 + 3 + 1 + 1 = 6
output dims. We therefore implement Linear(1536, 256) → GELU → Dropout(0.1)
→ Linear(256, 6) and split the 6-wide output as:
    [:, 0:1] -> log_vo
    [:, 1:4] -> charge_logits   (CE target in Stage 1)
    [:, 4:5] -> log_K_eq
    [:, 5:6] -> delta_Ef
This is the only deviation from the literal doc width (4 -> 6) and is
documented here and inline.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Hyperparameters (design doc §3.6) ─────────────────────────────────────────
PHYSICS_LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PHYSICS_HIDDEN_SIZE = 1536       # Qwen2.5-1.5B hidden size
Z_PHYS_DIM = 32                  # z_phys embedding dim (§3.6.5)
MAX_SEQ_LEN = 2048               # §3.6.1
NUMERIC_HEAD_HIDDEN = 256        # §3.6.4
NUMERIC_HEAD_OUT = 6             # 1 (log_vo) + 3 (charge) + 1 (log_K_eq) + 1 (dEf)
NUMERIC_HEAD_DROPOUT = 0.1       # §3.6.4

# LoRA config (§3.6.2)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class NumericPriorHead(nn.Module):
    """
    §3.6.4 numeric prior head: Linear(1536,256) → GELU → Dropout → Linear(256,6).

    The 6-wide output is split into the four semantic predictions documented
    in the module docstring.
    """

    def __init__(
        self,
        in_dim: int = PHYSICS_HIDDEN_SIZE,
        hidden_dim: int = NUMERIC_HEAD_HIDDEN,
        dropout: float = NUMERIC_HEAD_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUMERIC_HEAD_OUT),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pooled: [B, in_dim] mean-pooled last hidden state.

        Returns:
            dict with keys:
              - ``log_vo``       : [B, 1]
              - ``charge_logits``: [B, 3]  (CE over V_O charge 0/+1/+2)
              - ``log_K_eq``     : [B, 1]
              - ``delta_Ef``     : [B, 1]
        """
        out = self.net(pooled)  # [B, 6]
        return {
            "log_vo": out[:, 0:1],
            "charge_logits": out[:, 1:4],
            "log_K_eq": out[:, 4:5],
            "delta_Ef": out[:, 5:6],
        }


class _TinyToyTransformer(nn.Module):
    """
    2-layer toy causal transformer used ONLY by the CPU self-test
    (``tiny_test=True``). It mimics the parts of a HF causal LM that
    ``PhysicsLLM`` touches: an ``embed_tokens`` layer and a forward that
    returns ``output.hidden_states[-1]`` of shape [B, T, hidden].

    This avoids loading the real 1.5B weights for a shape/wiring test.
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
        out.hidden_states = (h,)  # only last layer needed
        return out


class PhysicsLLM(nn.Module):
    """
    Physics-LLM (M6): QLoRA-wrapped Qwen2.5-1.5B + z_phys projection +
    numeric prior head.

    Build either:
      - ``PhysicsLLM(tiny_test=True)`` — toy backbone, CPU, no downloads
        (used by the self-test and any CI shape check); or
      - ``PhysicsLLM.from_pretrained_qlora(...)`` — real 4-bit QLoRA model.

    Args:
        backbone   : the language model (HF PEFT model, or toy transformer).
                     ``extract_z_phys`` calls it with ``output_hidden_states``.
        tokenizer  : HF tokenizer (None in tiny_test mode).
        hidden_size: backbone hidden size (1536 for Qwen2.5-1.5B).
        max_len    : tokenizer truncation length (§3.6.1, default 2048).
        tiny_test  : if True, build a 2-layer toy transformer instead.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        tokenizer=None,
        hidden_size: int = PHYSICS_HIDDEN_SIZE,
        max_len: int = MAX_SEQ_LEN,
        tiny_test: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.tiny_test = tiny_test
        self.tokenizer = tokenizer

        if tiny_test:
            # small hidden for a fast CPU test; vocab 256 to match toy ids
            self.backbone = backbone or _TinyToyTransformer(hidden=hidden_size)
        else:
            if backbone is None:
                raise ValueError(
                    "PhysicsLLM requires a backbone unless tiny_test=True. "
                    "Use PhysicsLLM.from_pretrained_qlora(...) for the real model."
                )
            self.backbone = backbone

        # z_phys projection: 1536 -> 32, bias-free (§3.6.5). Trained in Stage 1,
        # frozen in Stage 3.
        self.z_proj = nn.Linear(hidden_size, Z_PHYS_DIM, bias=False)

        # numeric prior head (§3.6.4). Trained in Stage 1, frozen in Stage 3.
        self.numeric_head = NumericPriorHead(in_dim=hidden_size)

    # ── Embedding extraction (§3.6.5 / mean-pool of §3.5.4 style) ─────────────

    def _pooled_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean-pool of the last hidden state over non-pad tokens."""
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

    def extract_z_phys(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize ``texts`` (max_len 2048), forward, masked mean-pool the last
        hidden state, then project 1536 → 32.

        Args:
            texts: list of physics-reasoning prompts (length B).

        Returns:
            z_phys: [B, 32].
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "extract_z_phys needs a tokenizer; build via from_pretrained_qlora."
            )
        enc = self.tokenizer(
            texts, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding=True,
        )
        device = next(self.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pooled = self._pooled_hidden(input_ids, attention_mask)
        return self.z_proj(pooled)

    def forward_from_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Token-level entry point (used by the self-test and Stage-1 training,
        which pre-tokenizes). Returns ``(z_phys, numeric_dict)``.
        """
        pooled = self._pooled_hidden(input_ids, attention_mask)
        z_phys = self.z_proj(pooled)
        numerics = self.numeric_head(pooled)
        return z_phys, numerics

    def numeric_prior(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Convenience: tokenize → pooled → numeric prior head outputs."""
        if self.tokenizer is None:
            raise RuntimeError("numeric_prior needs a tokenizer.")
        enc = self.tokenizer(
            texts, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding=True,
        )
        device = next(self.parameters()).device
        pooled = self._pooled_hidden(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        return self.numeric_head(pooled)

    # ── Stage-3 freezing (H2 parameter-budget constraint, §1.4) ───────────────

    def n_trainable(self) -> int:
        """Number of trainable (requires_grad=True) parameters across the whole module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_trainable_breakdown(self) -> dict[str, int]:
        """Trainable param count split into LoRA / projection / numeric head."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        lora = sum(
            p.numel() for n, p in self.backbone.named_parameters()
            if p.requires_grad and "lora_" in n.lower()
        )
        return {
            "lora": lora,
            "z_proj": _count(self.z_proj),
            "numeric_head": _count(self.numeric_head),
            "total": self.n_trainable(),
        }

    def freeze_for_stage3(self) -> None:
        """
        Freeze ALL params (LoRA, projection, numeric head) and switch to eval.

        After this call ``n_trainable()`` must be 0 — M6 contributes no
        trainable params to the downstream V_O/PDR regression (§1.4 H2).
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ── Real model builder (guarded so import works without a GPU) ────────────

    @classmethod
    def from_pretrained_qlora(
        cls,
        model_name: str = PHYSICS_LLM_NAME,
        device_map: str | dict | None = "auto",
        max_len: int = MAX_SEQ_LEN,
        gradient_checkpointing: bool = True,
    ) -> "PhysicsLLM":
        """
        Build the real 4-bit (NF4) QLoRA Physics-LLM.

        Imports of bitsandbytes / transformers / peft happen INSIDE this method
        so the module imports cleanly on a GPU-less machine.

        Args:
            model_name           : HF model id (default Qwen2.5-1.5B-Instruct).
            device_map           : passed to ``from_pretrained`` (e.g. "auto",
                                   ``{"": "cuda:1"}``).
            max_len              : tokenizer truncation length (§3.6.1).
            gradient_checkpointing: enable for Stage-1 memory savings.

        Returns:
            A ``PhysicsLLM`` wrapping the PEFT-wrapped 4-bit backbone.
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

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(base, lora_config)

        hidden_size = getattr(peft_model.config, "hidden_size", PHYSICS_HIDDEN_SIZE)
        return cls(
            backbone=peft_model,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            max_len=max_len,
            tiny_test=False,
        )


def _estimate_lora_params(
    hidden: int, n_layers: int, r: int,
    intermediate: int, target_modules: list[str],
) -> int:
    """
    Analytic LoRA trainable-param estimate for the real Qwen2.5-1.5B
    (used only to print an expectation in the self-test; the toy backbone
    has no LoRA). For a Linear(in,out), LoRA adds r*(in+out) params.
    Qwen2.5-1.5B: hidden=1536, n_layers=28, intermediate=8960,
    GQA -> k_proj/v_proj out = 256 (2 KV heads * 128 head_dim).
    """
    kv_dim = 256  # Qwen2.5-1.5B GQA KV projection out dim
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
    """CPU self-test with a TINY toy backbone (no 1.5B load, no GPU)."""
    torch.manual_seed(0)

    # Build with a deliberately small hidden so the toy transformer is fast,
    # but keep the projection/head wired at the real semantics.
    tiny_hidden = 64
    model = PhysicsLLM(tiny_test=True, hidden_size=tiny_hidden)

    print("[PhysicsLLM self-test]")
    proj = sum(p.numel() for p in model.z_proj.parameters())
    head = sum(p.numel() for p in model.numeric_head.parameters())
    print(f"  z_proj params (tiny hidden={tiny_hidden})       : {proj:,}")
    print(f"  numeric_head params (tiny hidden={tiny_hidden}) : {head:,}")

    # Report REAL-config head/projection sizes (hidden=1536) for the report.
    real_proj = PHYSICS_HIDDEN_SIZE * Z_PHYS_DIM  # bias-free
    real_head = (
        (PHYSICS_HIDDEN_SIZE + 1) * NUMERIC_HEAD_HIDDEN
        + (NUMERIC_HEAD_HIDDEN + 1) * NUMERIC_HEAD_OUT
    )
    print(f"  [real cfg] z_proj (1536->32, no bias)     : {real_proj:,}")
    print(f"  [real cfg] numeric_head (1536->256->6)    : {real_head:,}")
    lora_est = _estimate_lora_params(
        hidden=PHYSICS_HIDDEN_SIZE, n_layers=28, r=LORA_R,
        intermediate=8960, target_modules=LORA_TARGET_MODULES,
    )
    print(f"  [real cfg] LoRA est (r=8, 28 layers)      : {lora_est:,} (~{lora_est/1e6:.2f} M)")

    # forward via token ids on the toy backbone
    B, T = 4, 16
    input_ids = torch.randint(0, 256, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, T // 2:] = 0  # exercise padding mask

    z_phys, numerics = model.forward_from_ids(input_ids, attention_mask)
    assert z_phys.shape == (B, Z_PHYS_DIM), f"z_phys shape {z_phys.shape}"
    assert numerics["log_vo"].shape == (B, 1)
    assert numerics["charge_logits"].shape == (B, 3)
    assert numerics["log_K_eq"].shape == (B, 1)
    assert numerics["delta_Ef"].shape == (B, 1)
    assert torch.isfinite(z_phys).all()
    print(f"  extract z_phys           : {tuple(z_phys.shape)} (OK)")
    print(f"  numeric split shapes     : "
          f"log_vo{tuple(numerics['log_vo'].shape)} "
          f"charge{tuple(numerics['charge_logits'].shape)} "
          f"log_K_eq{tuple(numerics['log_K_eq'].shape)} "
          f"dEf{tuple(numerics['delta_Ef'].shape)} (OK)")

    # gradient flows to projection + numeric head
    loss = z_phys.pow(2).sum() + sum(v.pow(2).sum() for v in numerics.values())
    loss.backward()
    assert model.z_proj.weight.grad is not None, "no grad to z_proj"
    print(f"  trainable (toy)          : {model.n_trainable():,}")

    # freeze_for_stage3 drops trainable to 0
    model.freeze_for_stage3()
    assert model.n_trainable() == 0, "freeze_for_stage3 left trainable params"
    assert not model.training
    print(f"  after freeze             : trainable={model.n_trainable()} eval={not model.training}")
    print("[PhysicsLLM self-test] PASS")


if __name__ == "__main__":
    _self_test()
