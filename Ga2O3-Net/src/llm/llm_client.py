"""LLMClient abstraction layer for V56.

V55-Ext hardcoded LLM calls to QwenChat. V56-Ext-2 needs to support:
  - Qwen2.5-7B-Instruct (text, ~14GB fp16)        — Stage 1 Triage, Stage 5 secondary
  - Qwen2.5-14B-Instruct-AWQ (text, ~9GB int4)    — Stage 2-T1/T2, Stage 5 primary
  - Qwen2.5-Math-7B-Instruct (text, ~14GB fp16)  — V56-SR-Evolve explore island
  - Qwen2.5-VL-7B-Instruct (vision, ~16GB fp16)  — Stage 3 figure extraction
  - HostedAPIClient (stub)                        — raises NoApiKeyError; Phase 56 zero-API

This module provides:
  - `LLMClient` Protocol — abstract chat + extract_structured
  - `QwenClient` — wraps existing QwenChat
  - `QwenAWQClient` — loads AWQ models via transformers + autoawq
  - `QwenVisionClient` — wraps existing QwenVision
  - `HostedAPIClient` — stub raises NoApiKeyError
  - `make_client(backend, ...)` — factory

Schema enforcement is Pydantic-first (vs Phase 55 regex). If `outlines`
is available, can additionally constrain decoding; otherwise we fall
back to retry-on-validation-error pattern (same robustness as Phase 55,
but with type safety).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.qwen_local import (
    QwenChat,
    QwenVision,
    _DEFAULT_CHAT_PATH,
    _DEFAULT_VISION_PATH,
    _DEFAULT_MATH_PATH,
    _extract_first_json,
    _sha8,
)


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_AWQ_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen2.5-14b-awq"

# V57 — Qwen3 MoE backbones (AWQ-4bit, ~18-19GB each)
_DEFAULT_QWEN3_THINKING_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen3-30b-a3b-thinking-awq"
_DEFAULT_QWEN3_INSTRUCT_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen3-30b-a3b-instruct-awq"
_DEFAULT_QWEN3_VL_PATH       = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen3-vl-30b-instruct-awq"

_DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


class NoApiKeyError(RuntimeError):
    pass


# ============================================================
# Protocol
# ============================================================


class LLMClient(Protocol):
    """Abstract LLM interface for V56 pipelines."""

    name: str

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        """Raw text completion."""
        ...

    def extract_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        """Chat and parse → Pydantic model. Returns None if validation fails."""
        ...


# ============================================================
# Helpers
# ============================================================


def _chat_to_pydantic(
    raw_response: str,
    schema: Type[T],
) -> Optional[T]:
    """Parse raw LLM response → Pydantic model.

    Steps:
      1. Strip code-fence markdown wrappers
      2. Find first balanced JSON object/array
      3. Parse as JSON
      4. Validate against Pydantic schema
    """
    data = _extract_first_json(raw_response)
    if data is None:
        return None
    if isinstance(data, list) and data:
        data = data[0]  # take first if list
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        logger.debug(f"Pydantic validation failed for {schema.__name__}: {e}")
        return None


def _retry_with_validation_prompt(
    chat_fn,
    system: str,
    user: str,
    schema: Type[T],
    max_retries: int,
) -> Optional[T]:
    """Generic retry loop with schema reminder on failure."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    schema_hint = (
        f"\n\nIMPORTANT: Respond ONLY with valid JSON matching this schema "
        f"(no prose, no markdown):\n{schema_json}"
    )
    prompt = user + schema_hint

    for attempt in range(max_retries + 1):
        response = chat_fn(system, prompt)
        result = _chat_to_pydantic(response, schema)
        if result is not None:
            return result
        if attempt < max_retries:
            prompt = (
                f"{user}{schema_hint}\n\nYour previous response was not "
                f"valid JSON matching the schema. Output ONLY JSON, no other text."
            )
    logger.warning(
        f"extract_structured failed after {max_retries + 1} attempts; "
        f"schema={schema.__name__}, prompt_sha={_sha8(user)}"
    )
    return None


# ============================================================
# Qwen (text)
# ============================================================


class QwenClient:
    """Local Qwen2.5-7B-Instruct (or Math-7B) text client.

    Wraps existing QwenChat to add Pydantic schema enforcement.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_CHAT_PATH,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        self.name = name or f"qwen-7b@{Path(model_path).name}"
        self._chat = QwenChat(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        return self._chat.chat(system, user, max_new_tokens=max_new_tokens)

    def extract_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        return _retry_with_validation_prompt(
            self._chat.chat, system, user, schema, max_retries
        )


# ============================================================
# Qwen-AWQ (text, int4)
# ============================================================


class QwenAWQClient:
    """Qwen2.5-14B-Instruct-AWQ (int4) via transformers + autoawq.

    Loads on demand. ~9GB VRAM at int4 (vs 28GB fp16); leaves room for
    Qwen-7B on the same card. Inference is ~2-3× slower per token than
    fp16 but still 5-10 tok/s on RTX 3090.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_AWQ_PATH,
        device: str = "cuda:1",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        self.name = name or f"qwen-14b-awq@{Path(model_path).name}"
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen AWQ model not found at {self.model_path}. "
                f"Download with: hf download Qwen/Qwen2.5-14B-Instruct-AWQ "
                f"--local-dir {self.model_path}"
            )

        device_map = "auto" if self.device == "auto" else {"": self.device}
        logger.info(f"Loading Qwen-14B-AWQ from {self.model_path} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self._model.eval()
        torch.manual_seed(self.seed)
        logger.info("Qwen-14B-AWQ loaded.")

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        max_tok = max_new_tokens or self.max_new_tokens
        prompt_sha = _sha8(prompt_text)

        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tok,
                "do_sample": self.temperature > 0.0,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if self.temperature > 0.0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = self.top_p
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_sha = _sha8(response)
        logger.debug(
            f"AWQ chat prompt_sha={prompt_sha} response_sha={response_sha} "
            f"out_tok={len(gen_ids)}"
        )
        return response

    def extract_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        return _retry_with_validation_prompt(
            self.chat, system, user, schema, max_retries
        )


# ============================================================
# Qwen-Vision
# ============================================================


class QwenVisionClient:
    """Qwen2.5-VL-7B-Instruct for figure extraction (Stage 3)."""

    def __init__(
        self,
        model_path: str = _DEFAULT_VISION_PATH,
        device: str = "auto",
        max_new_tokens: int = 2048,
        name: Optional[str] = None,
    ):
        self.name = name or f"qwen-vl-7b@{Path(model_path).name}"
        self._vision = QwenVision(
            model_path=model_path, device=device, max_new_tokens=max_new_tokens,
        )

    def extract_from_image(self, image_path: str, prompt: str) -> str:
        return self._vision.extract_from_image(image_path, prompt)

    def extract_structured_from_image(
        self,
        image_path: str,
        prompt: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        schema_hint = (
            f"\n\nIMPORTANT: Respond ONLY with valid JSON matching schema:\n{schema_json}"
        )
        for attempt in range(max_retries + 1):
            response = self._vision.extract_from_image(image_path, prompt + schema_hint)
            result = _chat_to_pydantic(response, schema)
            if result is not None:
                return result
        return None


# ============================================================
# Hosted API (stub — Phase 56 zero-API)
# ============================================================


class HostedAPIClient:
    """Stub for V56 Roadmap recommendation. Phase 56 explicitly zero-API:
    user decision 2026-05-24 to continue Phase 55 local-Qwen mode.

    This client exists so V56 pipeline code is forward-compatible with
    a future hosted-API switch; instantiating it raises immediately.
    """

    name = "hosted-api-stub"

    def __init__(self, model: str = "hosted-model", **kwargs):
        api_key = os.environ.get("HOSTED_LLM_API_KEY")
        if not api_key:
            raise NoApiKeyError(
                "HOSTED_LLM_API_KEY not set. Phase 56 was configured for zero-API "
                "local-Qwen mode. To enable a hosted-API backbone, export the key "
                "and re-instantiate this client."
            )
        raise NotImplementedError(
            "HostedAPIClient is a stub. Implement only when an api-key is provisioned "
            "AND the user authorizes API spend."
        )

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        raise NotImplementedError()

    def extract_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        raise NotImplementedError()


# ============================================================
# V57 — vLLM HTTP client (Qwen3-30B-A3B Thinking/Instruct via TP=2)
# ============================================================


class VLLMClient:
    """Talks to a vLLM OpenAI-compatible server.

    Designed for Qwen3-30B-A3B (MoE) on dual RTX 3090 TP=2 with FlashInfer.
    The server is launched separately (see docker/v57-vllm-*.yaml). This
    client only sends HTTP requests, so it adds ~0 VRAM and can be
    instantiated alongside Unsloth training (provided the server is down).

    Thinking-mode caveat: Qwen3-Thinking emits <think>...</think> blocks;
    set `strip_think=True` (default) to drop them before schema validation.
    """

    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_VLLM_BASE_URL,
        api_key: str = "EMPTY",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 600.0,
        strip_think: bool = True,
        name: Optional[str] = None,
        # Compatibility kwargs from callers that target local model clients.
        # vLLM is HTTP-only so we ignore device / seed (vLLM server handles
        # these via its own --seed flag), but accept them so make_client(
        # "qwen3-30b-thinking-vllm", device=..., seed=...) doesn't break.
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **_ignored,
    ):
        self.name = name or f"vllm@{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.strip_think = strip_think

    def _strip_think(self, text: str) -> str:
        if not self.strip_think:
            return text
        # Drop <think>...</think> blocks if present (Qwen3-Thinking format)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens or self.max_new_tokens,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"vLLM {self.base_url} request failed: {e}") from e
        msg = r.json()["choices"][0]["message"]
        # deepseek_r1 reasoning parser splits Qwen3-Thinking output into
        # `reasoning` (the <think>...</think> chain) and `content` (the
        # post-thinking answer). If the reply got cut off mid-think the
        # content may be null — fall back to reasoning in that case so we
        # still see something parseable for the schema retry loop.
        text = msg.get("content")
        if not text:
            text = msg.get("reasoning") or ""
        return self._strip_think(text)

    def extract_structured(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_retries: int = 2,
    ) -> Optional[T]:
        return _retry_with_validation_prompt(
            self.chat, system, user, schema, max_retries
        )


# ============================================================
# Factory
# ============================================================


def make_client(
    backend: str,
    device: str = "auto",
    **kwargs,
) -> LLMClient:
    """Create an LLMClient by backend name.

    Args:
        backend: one of {"qwen-7b", "qwen-14b-awq", "qwen-math-7b",
                          "qwen-vl-7b", "qwen3-30b-thinking",
                          "qwen3-30b-instruct", "qwen3-vl",
                          "qwen3-30b-thinking-vllm", "qwen3-30b-instruct-vllm",
                          "hosted-api"}
        device: torch device ('cuda:0', 'cuda:1', 'auto')

    Returns:
        Instantiated LLMClient. For Qwen variants, lazy-loads on first chat.
        For vLLM variants, no model load happens here — server must be running.
    """
    backend = backend.lower()
    if backend == "qwen-7b":
        return QwenClient(model_path=_DEFAULT_CHAT_PATH, device=device, **kwargs)
    if backend == "qwen-math-7b":
        return QwenClient(model_path=_DEFAULT_MATH_PATH, device=device, **kwargs)
    if backend == "qwen-14b-awq":
        return QwenAWQClient(model_path=_DEFAULT_AWQ_PATH, device=device, **kwargs)
    if backend == "qwen-vl-7b":
        return QwenVisionClient(model_path=_DEFAULT_VISION_PATH, device=device, **kwargs)
    # V57 — Qwen3 MoE direct in-process (Unsloth / transformers AWQ load)
    if backend == "qwen3-30b-thinking":
        return QwenAWQClient(model_path=_DEFAULT_QWEN3_THINKING_PATH,
                              device=device, name="qwen3-30b-thinking", **kwargs)
    if backend == "qwen3-30b-instruct":
        return QwenAWQClient(model_path=_DEFAULT_QWEN3_INSTRUCT_PATH,
                              device=device, name="qwen3-30b-instruct", **kwargs)
    if backend == "qwen3-vl":
        return QwenVisionClient(model_path=_DEFAULT_QWEN3_VL_PATH,
                                 device=device, name="qwen3-vl", **kwargs)
    # V57 — vLLM HTTP (preferred for TP=2 30B inference)
    if backend == "qwen3-30b-thinking-vllm":
        return VLLMClient(model="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",
                          name="qwen3-30b-thinking-vllm", **kwargs)
    if backend == "qwen3-30b-instruct-vllm":
        return VLLMClient(model="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
                          name="qwen3-30b-instruct-vllm",
                          strip_think=False, **kwargs)
    if backend in ("hosted-api", "hosted-llm"):
        return HostedAPIClient(model=backend, **kwargs)
    raise ValueError(
        f"Unknown LLM backend: {backend}. Supported: qwen-7b, qwen-math-7b, "
        f"qwen-14b-awq, qwen-vl-7b, qwen3-30b-thinking, qwen3-30b-instruct, "
        f"qwen3-vl, qwen3-30b-thinking-vllm, qwen3-30b-instruct-vllm, "
        f"hosted-api"
    )
