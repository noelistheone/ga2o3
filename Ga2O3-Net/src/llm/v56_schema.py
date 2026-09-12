"""Pydantic schemas for V56-Ext-2 extraction pipeline.

These schemas enforce structural validity at extraction time (vs Phase 55
ad-hoc regex JSON parsing). Each schema corresponds to one V56-Ext-2 stage:

- TriageResult        — Stage 1 (binary classification)
- SputterRow          — Stage 2-T1 (explicit extraction)
- SputterRowDerived   — Stage 2-T2 (derived fields)
- FigureExtraction    — Stage 3 (Qwen2.5-VL output)
- FieldVerification   — Stage 4 (ChatExtract verification)
- ConsensusResult     — Stage 5 (multi-LLM agreement)

Field constraints encode physical plausibility:
- V_O_log10_cm3 in [14, 22] — β-Ga2O3 physical range
- PDR_log10 in [0, 8]
- temperatures in [20, 1500] °C
- dopant_at_pct in [0, 20] (typical sputter ranges)
"""
from __future__ import annotations

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, ConfigDict


DepositionMethod = Literal[
    "RF sputter", "DC sputter", "MOCVD", "HVPE", "PLD",
    "Mist-CVD", "sol-gel", "ALD", "PE-ALD", "other",
]

AnnealAtmosphere = Literal[
    "O2", "N2", "Ar", "air", "vacuum", "forming gas", "H2", "N2O",
]

VOMethod = Literal[
    "XPS O1s deconvolution", "Hall", "PL", "SIMS",
    "positron annihilation", "TSC", "EPR", "Raman", "other",
]

ConfidenceLevel = Literal["high", "medium", "low"]


class TriageResult(BaseModel):
    """Stage 1 binary triage. Drops ~60-70% of PDFs that aren't relevant."""

    model_config = ConfigDict(extra="forbid")

    is_sputter_Ga2O3: bool = Field(
        ..., description="Paper reports β-Ga2O3 deposited by sputtering"
    )
    has_PDR: bool = Field(
        ..., description="Paper reports photo-dark ratio or photocurrent/dark current"
    )
    has_V_O: bool = Field(
        ..., description="Paper reports oxygen vacancy concentration"
    )
    has_dopant: bool = Field(
        ..., description="Paper studies a doped Ga2O3 film (vs. undoped)"
    )
    reason: str = Field(
        ..., min_length=10, max_length=300,
        description="One-sentence justification for the classification",
    )

    def passes(self) -> bool:
        """Pass if sputter-Ga2O3 AND at least one measurement is reported."""
        return self.is_sputter_Ga2O3 and (self.has_PDR or self.has_V_O)


class SputterRow(BaseModel):
    """Stage 2-T1 explicit extraction. One row per (paper, sample).

    Strict mode: no extra fields, all bounds physical.
    `quote` is the anti-hallucination anchor — must be verbatim from PDF.
    """

    model_config = ConfigDict(extra="forbid")

    dopant: Optional[str] = Field(
        None, description="Single element symbol (e.g. 'Sn'); None if undoped"
    )
    dopant_at_pct: Optional[float] = Field(
        None, ge=0, le=20,
        description="Dopant atomic percent (0 to 20)",
    )
    deposition_method: Optional[str] = Field(
        "other",
        description="Film deposition technique (free text; common: 'RF sputter', 'DC sputter', 'MOCVD', 'HVPE', 'PLD', 'Mist-CVD')",
    )
    substrate_temp_C: Optional[float] = Field(
        None, ge=20, le=1200,
        description="Substrate temperature during deposition (°C)",
    )
    O2_Ar_ratio: Optional[float] = Field(
        None, ge=0, le=10,
        description="O2:Ar gas flow ratio (0 = pure Ar, >1 = O2-rich)",
    )
    total_pressure_Pa: Optional[float] = Field(
        None, ge=1e-4, le=1e4,
        description="Total chamber pressure (Pa)",
    )
    sputter_power_W: Optional[float] = Field(
        None, ge=0, le=2000,
    )
    film_thickness_nm: Optional[float] = Field(
        None, ge=1, le=10000,
    )
    anneal_T_C: Optional[float] = Field(
        None, ge=20, le=1500,
        description="Post-deposition anneal temperature (°C)",
    )
    anneal_atm: Optional[str] = Field(
        None,
        description="Post-deposition anneal atmosphere (free text; common: 'O2', 'N2', 'Ar', 'air', 'vacuum', 'forming gas')",
    )
    anneal_time_min: Optional[float] = Field(
        None, ge=0, le=600,
    )
    substrate: Optional[str] = Field(
        None, description="Substrate material (e.g. 'c-sapphire', 'Si', 'GaN', 'native bulk')",
    )
    wavelength_nm: Optional[float] = Field(
        None, ge=100, le=1100,
        description="UV illumination wavelength (nm)",
    )
    bias_V: Optional[float] = Field(
        None, ge=-50, le=200,
        description="Applied bias voltage during measurement (V)",
    )
    PDR_log10: Optional[float] = Field(
        None, ge=-1, le=8,
        description="log10(photo current / dark current). β-Ga2O3 typical 2-6.",
    )
    photo_current_A: Optional[float] = Field(
        None, ge=1e-15, le=1.0,
    )
    dark_current_A: Optional[float] = Field(
        None, ge=1e-15, le=1.0,
    )
    responsivity_AW: Optional[float] = Field(
        None, ge=0, le=1e6,
    )
    V_O_log10_cm3: Optional[float] = Field(
        None, ge=14, le=22,
        description="log10(oxygen vacancy concentration / cm^-3). Physical range 14-22.",
    )
    V_O_method: Optional[str] = Field(
        None,
        description="Technique used to estimate V_O (free text; common: 'XPS O1s deconvolution', 'Hall', 'PL', 'SIMS', 'positron annihilation')",
    )
    confidence: ConfidenceLevel = Field(
        "low",
        description="Overall extraction confidence (high → weight 1.0; med → 0.5; low → 0.0)",
    )
    quote: str = Field(
        "",
        max_length=1500,
        description="Verbatim sentence(s) from PDF supporting the extraction (anti-hallucination)",
    )


class SputterRowDerived(BaseModel):
    """Stage 2-T2 derived fields. LLM is allowed to infer null fields
    from I-V/anneal hints; must cite quote span supporting the inference."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Name of derived field in SputterRow")
    value: Optional[float] = Field(None)
    derivation: str = Field(
        ..., min_length=10,
        description="Step-by-step derivation from quoted source span",
    )
    source_quote: str = Field(
        ..., min_length=20, max_length=1500,
    )


class FigureExtraction(BaseModel):
    """Stage 3 Qwen2.5-VL figure digitization output.

    For an I-V curve figure, returns (V, I_dark, I_photo) tuples and
    derived PDR/responsivity at the illumination wavelength.
    """

    model_config = ConfigDict(extra="forbid")

    figure_id: str
    figure_type: Literal["I-V", "I-t", "responsivity-V", "EQE-lambda", "other"]
    wavelength_nm: Optional[float] = Field(None, ge=100, le=1100)
    data_points: List[List[float]] = Field(
        ..., description="List of [V, I_dark, I_photo] or [t, I] tuples",
    )
    derived_PDR_log10: Optional[float] = Field(None, ge=-1, le=8)
    confidence: ConfidenceLevel
    reasoning: str = Field(..., min_length=10, max_length=600)


class FieldVerification(BaseModel):
    """Stage 4 ChatExtract verification. For each field in SputterRow,
    we ask 'Is this value certainly X?' and gate on the response."""

    model_config = ConfigDict(extra="forbid")

    field: str
    claimed_value: Optional[float] = None
    claimed_value_str: Optional[str] = None
    verdict: Literal["yes", "no", "uncertain"]
    reason: str = Field(..., min_length=5, max_length=400)


class ConsensusResult(BaseModel):
    """Stage 5 multi-LLM consensus. Computed from Stage 2-T1 outputs of
    two independent LLMs (Qwen-7B and Qwen-14B-AWQ in our config)."""

    model_config = ConfigDict(extra="forbid")

    fields_compared: int
    fields_agreed: int
    agreement_rate: float = Field(..., ge=0, le=1)
    numeric_within_10pct: int = Field(
        ..., description="Count of numeric fields within 10% tolerance"
    )
    confidence_weight: float = Field(
        ..., ge=0, le=1,
        description="Final confidence_weight = agreement_rate × redundancy_pass_rate",
    )


CONFIDENCE_TO_WEIGHT: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.0}


def confidence_to_sample_weight(level: ConfidenceLevel) -> float:
    """Map V56 confidence Literal → numeric sample_weight for trainer."""
    return CONFIDENCE_TO_WEIGHT[level]


# ============================================================
# V57 — multimodal extraction extension
# ============================================================
# These schemas extend V56 for V57-Ext-3 (zero-API path). New stages:
#   TriageResultVL          — VL-aware triage (flags XPS/IV figure pages)
#   XPSPeak / XPSDeconvolution — O 1s deconvolution → derived [V_O]
#   HSE06PlausibilityVerdict   — Qwen3-30B-Thinking validator against HSE06 window


class TriageResultVL(TriageResult):
    """VL-aware triage; extends V56 TriageResult with figure page indices."""

    has_O1s_XPS_figure: bool = Field(
        False, description="Paper contains an O 1s XPS deconvolution figure"
    )
    has_IV_figure: bool = Field(
        False, description="Paper contains an I-V or I-t curve under UV"
    )
    xps_figure_pages: List[int] = Field(
        default_factory=list,
        description="1-indexed page numbers containing O 1s XPS figures",
    )
    iv_figure_pages: List[int] = Field(
        default_factory=list,
        description="1-indexed page numbers containing I-V/I-t curves",
    )


class XPSPeak(BaseModel):
    """Single deconvolved component in an O 1s XPS spectrum."""

    model_config = ConfigDict(extra="forbid")

    binding_energy_eV: float = Field(
        ..., ge=526, le=536,
        description="O 1s component binding energy (typical 528-534 eV)",
    )
    fraction: float = Field(
        ..., ge=0, le=1,
        description="Fractional area of this peak in the total O 1s envelope",
    )
    assignment: Literal["lattice_O", "O_v_or_OH", "adsorbed_O", "other"] = Field(
        ...,
        description="Authors' chemical assignment. NB: 531-532 eV is NOT proof of V_O alone.",
    )


class XPSDeconvolution(BaseModel):
    """Stage 2 V57-Ext-3 output: O 1s peak fit + derived V_O concentration.

    The `cation_valence_cross_validation` flag captures the Spectroscopy
    Online 2024 warning that 531-532 eV is often misassigned to V_O without
    cation-side XPS evidence. Used downstream to penalize confidence.
    """

    model_config = ConfigDict(extra="forbid")

    peaks: List[XPSPeak] = Field(..., min_length=1, max_length=6)
    derived_V_O_log10_cm3: Optional[float] = Field(
        None, ge=14, le=22,
        description="Authors' or model's [V_O] estimate (log10 cm^-3) if computable",
    )
    cation_valence_cross_validation: bool = Field(
        False,
        description="True if authors cross-validate via cation valence shift (Ga 2p/3d)",
    )
    quote: str = Field(
        ..., min_length=20, max_length=1500,
        description="Verbatim caption / paragraph from PDF supporting the extraction",
    )
    confidence: ConfidenceLevel = Field("low")


class HSE06PlausibilityVerdict(BaseModel):
    """Stage 3 V57-Ext-3 validator. Qwen3-30B-Thinking compares the derived
    V_O against the per-dopant HSE06 plausibility window computed from the
    87 HSE06 records (V57 DFT corpus).
    """

    model_config = ConfigDict(extra="forbid")

    field_in_hse06_window: bool = Field(
        ...,
        description="True if the derived V_O log10 is within the HSE06-derived window",
    )
    verdict: Literal["accept", "reject", "uncertain"] = Field(...)
    reasoning_trace: str = Field(
        ..., min_length=20, max_length=4000,
        description="Brief reasoning (think tokens stripped before storage)",
    )


# ============================================================
# V57 — Coscientist (inverse design) recipe schema
# ============================================================


class CoscientistRecipe(BaseModel):
    """One V57-Coscientist-DFT proposed recipe targeting PDR > 10^5.

    Each recipe must cite a specific HSE06 formation energy from the DFT
    evidence table; the audit script verifies cited E_f against the corpus.
    """

    model_config = ConfigDict(extra="forbid")

    recipe_id: int = Field(..., ge=1, le=10)
    dopant: str = Field(..., description="Primary dopant element symbol")
    dopant_at_pct: float = Field(..., ge=0, le=10)
    codopant: Optional[str] = Field(None)
    codopant_at_pct: Optional[float] = Field(None, ge=0, le=10)
    deposition_method: str = Field("RF sputter")
    substrate_temp_C: float = Field(..., ge=20, le=1000)
    O2_fraction: float = Field(..., ge=0, le=1, description="O2 / (O2+Ar) fraction")
    anneal_T_C: Optional[float] = Field(None, ge=20, le=1200)
    anneal_atm: Optional[str] = Field(None)
    anneal_time_min: Optional[float] = Field(None, ge=0, le=600)
    predicted_pdr_log10: float = Field(..., ge=0, le=8)
    predicted_V_O_log10: float = Field(..., ge=14, le=22)
    cited_hse06_record_ids: List[str] = Field(
        ..., min_length=1,
        description="Record ids from V_O_HSE06.yaml that justify this choice",
    )
    physics_reasoning: str = Field(..., min_length=50, max_length=4000)
