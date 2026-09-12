"""Local LLM stack — Qwen2.5 wrappers + LLMClient abstraction.

Phase 55 (V55-Ext/SR/AL/ICL/Coscientist): QwenChat / QwenVision / mock_chat
Phase 56 (V56-Ext-2 / V56-SR-Evolve / V56-Hypo / etc.): LLMClient Protocol,
  QwenClient, QwenAWQClient (Qwen-14B-Int4), HostedAPIClient (stub).
"""
from .qwen_local import QwenChat, QwenVision, mock_chat
from .llm_client import (
    LLMClient,
    QwenClient,
    QwenAWQClient,
    QwenVisionClient,
    HostedAPIClient,
    NoApiKeyError,
    make_client,
)
from .v56_schema import (
    TriageResult,
    SputterRow,
    SputterRowDerived,
    FigureExtraction,
    FieldVerification,
    ConsensusResult,
    CONFIDENCE_TO_WEIGHT,
    confidence_to_sample_weight,
)

__all__ = [
    "QwenChat", "QwenVision", "mock_chat",
    "LLMClient", "QwenClient", "QwenAWQClient", "QwenVisionClient",
    "HostedAPIClient", "NoApiKeyError", "make_client",
    "TriageResult", "SputterRow", "SputterRowDerived",
    "FigureExtraction", "FieldVerification", "ConsensusResult",
    "CONFIDENCE_TO_WEIGHT", "confidence_to_sample_weight",
]
