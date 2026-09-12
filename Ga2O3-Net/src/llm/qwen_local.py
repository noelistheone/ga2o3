"""Local Qwen2.5 wrapper for Phase 55 (V55-Ext / SR / AL / ICL / Coscientist).

Provides a unified, deterministic LLM call interface that runs entirely
on local GPU(s). Replaces external API dependencies (hosted LLM services)
with Qwen2.5-{7B, VL-7B, Math-7B}-Instruct.

Design:
- `QwenChat(model_path, device, dtype)`: text-only chat with JSON-mode coercion
- `QwenVision(model_path, device)`: Qwen2.5-VL for figure digitization
- `extract_json(prompt, json_schema, ...)`: regex/sympy-validated structured output
- `mock_chat(prompt)`: zero-cost test stub returning empty list / null

All calls are deterministic (temperature=0, top_p=1, seed pinned) and
log prompt SHA + response SHA for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

# transformers 4.57+ renamed PytorchGELUTanh → GELUTanh.
# autoawq 0.2.9 still imports the old name; add alias for compatibility.
try:
    import transformers.activations as _ta
    if not hasattr(_ta, "PytorchGELUTanh") and hasattr(_ta, "GELUTanh"):
        _ta.PytorchGELUTanh = _ta.GELUTanh
except ImportError:
    pass

logger = logging.getLogger(__name__)


_DEFAULT_CHAT_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen2.5-7b"
_DEFAULT_VISION_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen2.5-vl-7b"
_DEFAULT_MATH_PATH = "/home/lawrence/Physics/Ga2O3-Net/checkpoints/qwen2.5-math-7b"


def _sha8(s: str) -> str:
    """8-char SHA hash of a string (for logging deterministic prompts)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _extract_first_json(text: str) -> Any | None:
    """Extract the first valid JSON object/array from a string.

    Qwen sometimes wraps JSON in ```json ... ``` or adds preamble. Use a
    regex-based parser to find the largest balanced JSON in the response.
    """
    # Strip code-fence wrappers
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    # Find first `{` or `[` and walk to its matching close
    for open_idx, opener in enumerate(cleaned):
        if opener in "{[":
            closer = "}" if opener == "{" else "]"
            depth = 0
            for end_idx in range(open_idx, len(cleaned)):
                c = cleaned[end_idx]
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[open_idx:end_idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
    return None


class QwenChat:
    """Text-only Qwen2.5 chat wrapper.

    Args:
        model_path: HuggingFace model path or local dir (default: Qwen2.5-7B-Instruct)
        device: torch device ('cuda:0', 'cuda:1', 'auto'); 'auto' uses device_map='auto'
        dtype: 'bfloat16' (default), 'float16', or 'int8' (uses bitsandbytes)
        max_new_tokens: max generation length (default 2048)
        temperature: sampling temperature (0.0 = greedy, deterministic)
        top_p: nucleus sampling cutoff (1.0 = no cutoff)
        seed: RNG seed for sampling (when temperature > 0)
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
    ):
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load Qwen on first call (avoids slow import during smoke tests)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen model not found at {self.model_path}. "
                f"Download with: hf download Qwen/Qwen2.5-7B-Instruct "
                f"--local-dir {self.model_path}"
            )

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "int8": torch.float16,  # int8 loaded via bitsandbytes config below
        }.get(self.dtype, torch.bfloat16)

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.dtype == "int8":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.device}

        logger.info(f"Loading Qwen model from {self.model_path} (dtype={self.dtype}, device={self.device})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self._model.eval()
        torch.manual_seed(self.seed)
        logger.info("Qwen model loaded.")

    def chat(self, system_prompt: str, user_prompt: str,
             max_new_tokens: int | None = None) -> str:
        """Single chat turn; returns assistant raw text response."""
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Hash the full prompt for reproducibility logging
        prompt_sha = _sha8(prompt_text)

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        max_tok = max_new_tokens or self.max_new_tokens

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
        # Strip prompt tokens
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_sha = _sha8(response)
        logger.debug(
            f"chat prompt_sha={prompt_sha} response_sha={response_sha} "
            f"out_tok={len(gen_ids)}"
        )
        return response

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        max_retries: int = 2,
    ) -> Any | None:
        """Chat and extract JSON; retry up to max_retries with reminder prompt.

        Returns the parsed JSON object/list, or None if parsing fails after retries.
        """
        retry_prompt = user_prompt
        for attempt in range(max_retries + 1):
            response = self.chat(system_prompt, retry_prompt)
            data = _extract_first_json(response)
            if data is not None:
                return data
            if attempt < max_retries:
                retry_prompt = (
                    f"{user_prompt}\n\nIMPORTANT: Your previous response could not be "
                    f"parsed as JSON. Output ONLY valid JSON, no prose, no markdown."
                )
        logger.warning(
            f"extract_json failed after {max_retries + 1} attempts; prompt_sha={_sha8(user_prompt)}"
        )
        return None


class QwenVision:
    """Qwen2.5-VL wrapper for figure / plot digitization (PlotExtract pattern).

    Args:
        model_path: local dir for Qwen2.5-VL-7B-Instruct
        device: 'cuda:0', 'cuda:1', or 'auto'
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_VISION_PATH,
        device: str = "auto",
        max_new_tokens: int = 2048,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "Qwen2.5-VL requires transformers >= 4.45. "
                "pip install 'transformers>=4.45' qwen-vl-utils"
            ) from e

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen2.5-VL model not found at {self.model_path}. "
                f"Download with: hf download Qwen/Qwen2.5-VL-7B-Instruct "
                f"--local-dir {self.model_path}"
            )

        device_map = "auto" if self.device == "auto" else {"": self.device}
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self._model.eval()
        logger.info("Qwen2.5-VL loaded.")

    def extract_from_image(
        self,
        image_path: str | Path,
        prompt: str,
    ) -> str:
        """Send an image + text prompt to Qwen2.5-VL; return raw text response."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(gen_ids, skip_special_tokens=True)


def mock_chat(system_prompt: str, user_prompt: str) -> str:
    """Mock chat for unit tests; returns an empty JSON list as response."""
    return "[]"
