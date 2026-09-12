"""
Phase 30 — Differentiable Brouwer defect-chemistry head for log[V_O].

Replaces the scalar VC prediction head with a closed-form defect-equilibrium
expression. The fused embedding only predicts a small set of latent variables
(formation-energy residual, log-prefactor, dopant charge-balance term); the
final output is computed by the Brouwer equation, so temperature and oxygen-
partial-pressure dependencies are *guaranteed* by the formula rather than
learned from data.

Equation (oxygen vacancy in β-Ga₂O₃, neutral V_O for simplicity):
    O_O ⇌ V_O + ½ O₂(g) + 2e⁻

Boltzmann equilibrium:
    [V_O] · p(O₂)^(½) = K(T) · exp(−E_f^VO / kT)

Taking log₁₀:
    log₁₀[V_O] = log_prefactor
                 − E_f^VO / (kT · ln10)
                 − ½ · log₁₀(p_O₂)
                 + dopant_term

Notes
-----
* Targets are *standardized* log₁₀(V_O), so a learnable ``global_bias`` absorbs
  the absolute reference shift introduced by standardization. The physical
  *shape* (T sensitivity, pO₂ sensitivity, dopant-direction sensitivity)
  remains fixed by the closed form.
* ΔE_f is bounded to ±0.15 eV via tanh so the Boltzmann derivative does not
  saturate gradient flow at typical sputter temperatures (T~800 K → 1/(kT·ln10)
  ≈ 6.3 per eV; ΔE_f swing of ±0.15 eV gives ≈ ±0.95 in log₁₀[V_O], on the same
  order as standardized target σ).
* The head ignores E_f charge-state dependence on Fermi level; that physics is
  shoved into the trainable ``dopant_term`` which the latent net learns from
  the fused embedding (donor-rich → suppress V_O; acceptor-rich → enhance).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5
LN10 = math.log(10.0)


class BrouwerHeadVC(nn.Module):
    """Differentiable Brouwer-form VC head.

    Args:
        in_dim:        fused embedding size (typically 160).
        hidden_dim:    hidden size of the latent MLP.
        dropout:       dropout in the latent MLP.
        e_f_baseline:  reference V_O formation energy [eV] (typical 1.5–3 eV
                       for β-Ga₂O₃; 2.0 eV is a conservative midpoint).
        e_f_swing:     magnitude of the tanh-bounded ΔE_f around baseline.

    Forward:
        fused:    [B, in_dim]
        process:  [B, ≥3] — only columns 0 (T_C) and 2 (o2_fraction) are read.

    Returns:
        [B, 1] standardized log₁₀[V_O] estimate.
    """

    def __init__(
        self,
        in_dim: int = 160,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        e_f_baseline: float = 2.0,
        e_f_swing: float = 0.15,
        anchor_to_proxy: bool = False,
        use_method_offset: bool = False,
        n_methods: int = 6,
        use_class_offset: bool = False,
        n_classes: int = 4,
        use_conc_dep_dopant: bool = False,
        conc_ref_at_frac: float = 0.01,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.e_f_baseline = float(e_f_baseline)
        self.e_f_swing = float(e_f_swing)
        self.anchor_to_proxy = bool(anchor_to_proxy)
        self.use_method_offset = bool(use_method_offset)
        self.n_methods = int(n_methods)
        if self.use_method_offset:
            self.method_offset = nn.Embedding(self.n_methods, 1)
            nn.init.zeros_(self.method_offset.weight)
        # Phase 45 V10: per-valence-class scalar offset.
        # 4 classes — 0=acceptor (Mg/Zn/Cu, v=2), 1=isovalent (Al/Fe/B/V/Er/Eu, v=3),
        # 2=donor (Si/Sn/Ti/Ge, v=4), 3=super_donor (Sb/Ta/Bi/W, v≥5).
        # Initialised to zero; learned via main MSE alongside dopant_term.
        # Closes the Tier-1C gap on N=1 unsupervised elements (Ge/Bi/Sb FLIP
        # in V5) by giving every element-class a learnable shared baseline
        # offset that propagates to elements without VC labels through the
        # class-membership lookup. Lit anchor: Segal 2025 Transductive OOD
        # (npj Comp. Mat.) — predict relative to nearest-class anchor instead
        # of absolute, recovers extrapolation to unseen elements.
        self.use_class_offset = bool(use_class_offset)
        self.n_classes = int(n_classes)
        if self.use_class_offset:
            self.class_offset = nn.Embedding(self.n_classes, 1)
            nn.init.zeros_(self.class_offset.weight)

        # Phase 53 V53-γ1: concentration-dependent dopant_term.
        # Replaces the V52b scalar dopant_term with explicit log10([dopant])
        # dependence so within-DOI monotonicity is baked into the architecture
        # instead of imposed via soft hinge loss. Sign of slope is class-locked
        # (acceptor: -1, isovalent: 0, donor: +1, super_donor: +1) matching
        # LIT_RHO_VC empirical convention from eval_physics_diag_table.py.
        # When enabled, latent_net outputs 4 logits instead of 3:
        #   [ΔE_f raw, log_prefactor, dopant_b, dopant_a_logit]
        # and dopant_term = sign(class) · softplus(a_logit) · log10(c/c_ref) + b
        self.use_conc_dep_dopant = bool(use_conc_dep_dopant)
        self.conc_ref_at_frac = float(conc_ref_at_frac)
        # Sign LUT matches _ELEM_CLASS_MAP order (0=accep, 1=isov, 2=donor, 3=super_donor).
        # Stored as buffer so it moves with .to(device).
        self.register_buffer(
            "_class_sign_lut",
            torch.tensor([-1.0, 0.0, +1.0, +1.0], dtype=torch.float32),
        )

        latent_out_dim = 4 if self.use_conc_dep_dopant else 3
        self.latent_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_out_dim),
        )
        # Zero-init the output layer. Under V53-γ1: dopant_a_logit=0 →
        # softplus = ln(2) ≈ 0.69, so |a|≈0.69 from the start (sign-locked).
        # Phase 53 V53-γ1' (corrected): keep bias[3]=0, NOT -8.
        # The original -8 init created a saddle that the model never escaped:
        # softplus(-8)≈3e-4 → slope ≈ 0 → class-aware sign LUT was effectively
        # never used (Sn parity r dropped to -0.15, Within-element R² = -0.245).
        # bias[3]=0 gives softplus(0)·sign·log10(c/c_ref) as initial dopant_term,
        # which is meaningful but small (~±0.5 at typical doping). Trades the
        # "init exactly matches V52b" property for usable gradient flow.
        nn.init.zeros_(self.latent_net[-1].weight)
        nn.init.zeros_(self.latent_net[-1].bias)

        # Learnable bias absorbs target-standardization shift.
        self.global_bias = nn.Parameter(torch.zeros(1))

    def latent_terms(
        self,
        fused: torch.Tensor,
        dopant_specs: list[str] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Returns latents.

        - V52b path (use_conc_dep_dopant=False): (E_f, log_prefactor, dopant_term)
        - V53-γ1 path (use_conc_dep_dopant=True): (E_f, log_prefactor, dopant_b, dopant_a_logit)

        Each tensor is [B, 1].
        """
        latents = self.latent_net(fused)
        delta_ef = self.e_f_swing * torch.tanh(latents[:, 0:1])
        if self.anchor_to_proxy:
            if dopant_specs is None:
                raise ValueError(
                    "BrouwerHeadVC.anchor_to_proxy=True requires dopant_specs in forward"
                )
            from src.models.physics_features import compute_vo_formation_proxy
            baseline = compute_vo_formation_proxy(dopant_specs).to(fused.device).to(fused.dtype)
            baseline = baseline.unsqueeze(-1)  # [B, 1]
        else:
            baseline = torch.full_like(delta_ef, self.e_f_baseline)
        E_f = baseline + delta_ef
        log_prefactor = latents[:, 1:2]
        if self.use_conc_dep_dopant:
            dopant_b = latents[:, 2:3]
            dopant_a_logit = latents[:, 3:4]
            return E_f, log_prefactor, dopant_b, dopant_a_logit
        dopant_term = latents[:, 2:3]
        return E_f, log_prefactor, dopant_term

    def forward(
        self,
        fused: torch.Tensor,
        process: torch.Tensor,
        dopant_specs: list[str] | None = None,
        method_idx: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        element_offset: torch.Tensor | None = None,
        dopant_term_multiplier: torch.Tensor | None = None,
        dopant_total_conc: torch.Tensor | None = None,
        dopant_a_offset: torch.Tensor | None = None,
        dopant_b_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_conc_dep_dopant:
            E_f, log_prefactor, dopant_b, dopant_a_logit = self.latent_terms(
                fused, dopant_specs=dopant_specs
            )
            if dopant_total_conc is None:
                raise ValueError(
                    "BrouwerHeadVC.use_conc_dep_dopant=True requires dopant_total_conc"
                )
            if class_idx is None:
                raise ValueError(
                    "BrouwerHeadVC.use_conc_dep_dopant=True requires class_idx for sign LUT"
                )
            # Per-element bias from ElementHyperNet (out_dim=2 → split a_e, b_e).
            if dopant_a_offset is not None:
                dopant_a_logit = dopant_a_logit + dopant_a_offset
            if dopant_b_offset is not None:
                dopant_b = dopant_b + dopant_b_offset
            sign = self._class_sign_lut[class_idx].unsqueeze(-1)  # [B, 1]
            a = sign * torch.nn.functional.softplus(dopant_a_logit)
            log_c = torch.log10(
                dopant_total_conc.clamp(min=1e-6) / self.conc_ref_at_frac
            )
            dopant_term = a * log_c + dopant_b
        else:
            E_f, log_prefactor, dopant_term = self.latent_terms(
                fused, dopant_specs=dopant_specs
            )

        # Phase 47 V14: multiplicative element modulation on dopant_term.
        # Mutually exclusive with `element_offset` (additive). When provided,
        # dopant_term is scaled by (1 + dopant_term_multiplier) BEFORE the
        # closed-form Brouwer assembly. By construction H cannot collapse to
        # a constant (m=1.0 means H gives 0 gradient signal but the head
        # needs H to differentiate elements to fit the data).
        if dopant_term_multiplier is not None:
            dopant_term = dopant_term * (1.0 + dopant_term_multiplier)

        T_K = (process[:, 0:1] + 273.15).clamp(min=100.0)

        pO2 = process[:, 2:3].clamp(min=1e-4, max=1.0)
        log_pO2 = torch.log10(pO2)

        boltz = -E_f / (KB_EV * T_K * LN10)
        log_VO = log_prefactor + boltz - 0.5 * log_pO2 + dopant_term + self.global_bias

        if self.use_method_offset and method_idx is not None:
            log_VO = log_VO + self.method_offset(method_idx)
        # Phase 45 V10: per-valence-class scalar offset. class_idx is computed
        # in Ga2O3Net.forward via dopant-spec → cation → ELEM_VALENCE → class.
        if self.use_class_offset and class_idx is not None:
            log_VO = log_VO + self.class_offset(class_idx)
        # Phase 46 V11: per-element scalar offset from ElementHyperNet, computed
        # in Ga2O3Net.forward by mapping each row's dopant_specs to a [B, K, 5]
        # descriptor tensor + [B, K] normalized fractions and running the HN.
        # element_offset is [B, 1]. NOTE: under V53-γ1 (use_conc_dep_dopant=True)
        # the HN's output is split into a_e/b_e and applied inside dopant_term;
        # element_offset should be None in that path.
        if element_offset is not None:
            log_VO = log_VO + element_offset
        return log_VO


class BrouwerHeadPDR(nn.Module):
    """Phase 31 — Arrhenius-form closed head for log10[PDR].

    Closed form (Armstrong et al. arXiv 1812.07197 + Kublitski Nat Commun 2021
    SRH dark-current activation + Ohm's-law bias scaling):

        log10 PDR = log_prefactor
                    + E_a / (kT · ln10)         # Arrhenius dark-current activation
                    + log10(V_bias)             # bias-voltage scaling (Ohm)
                    + dopant_term               # free latent for chemistry
                    + global_bias               # absorbs target standardization

    Identical philosophy to ``BrouwerHeadVC``: hard-constrain only the
    *physics invariants* (T-Boltzmann direction, V-linear direction) and
    leave element-specific shifts to the unbounded ``dopant_term`` latent
    so the donor-vs-acceptor sign is learned from data, not imposed.

    Args:
        in_dim:        fused embedding size (typically 160).
        hidden_dim:    hidden size of the latent MLP.
        dropout:       dropout in the latent MLP.
        e_a_baseline:  reference dark-current activation energy [eV];
                       Armstrong 2018 reports 0.6–1.0 eV for Ga₂O₃ MSM.
        e_a_swing:     tanh-bounded ΔE_a around baseline. ±0.20 eV at
                       T=300 K gives ≈ ±3.4 in log10[PDR], the same scale
                       as a standardized target σ.

    Forward:
        fused:    [B, in_dim]
        process:  [B, ≥10] — reads col 0 (T_C) and col 9 (V_bias).

    Returns:
        [B, 1]  standardized log10[PDR] estimate.
    """

    def __init__(
        self,
        in_dim: int = 160,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        e_a_baseline: float = 0.7,
        e_a_swing: float = 0.20,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.e_a_baseline = float(e_a_baseline)
        self.e_a_swing = float(e_a_swing)

        self.latent_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # [ΔE_a raw, log_prefactor, dopant_term]
        )
        nn.init.zeros_(self.latent_net[-1].weight)
        nn.init.zeros_(self.latent_net[-1].bias)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def latent_terms(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (E_a, log_prefactor, dopant_term), each [B, 1]."""
        latents = self.latent_net(fused)
        delta_ea = self.e_a_swing * torch.tanh(latents[:, 0:1])
        E_a = self.e_a_baseline + delta_ea
        log_prefactor = latents[:, 1:2]
        dopant_term = latents[:, 2:3]
        return E_a, log_prefactor, dopant_term

    def forward(self, fused: torch.Tensor, process: torch.Tensor) -> torch.Tensor:
        E_a, log_prefactor, dopant_term = self.latent_terms(fused)

        # Temperature in K (clamp to avoid div-by-zero / negative kT).
        T_K = (process[:, 0:1] + 273.15).clamp(min=100.0)

        # Bias voltage. process[:, 9] is measurement_voltage_V (default 10 V).
        # Clamp to a 0.1 V floor — log10(0) is undefined.
        V = process[:, 9:10].clamp(min=0.1)
        log_V = torch.log10(V)

        # Closed form: Arrhenius dark + Ohm's-law photo bias scaling.
        boltz = E_a / (KB_EV * T_K * LN10)
        log_PDR = log_prefactor + boltz + log_V + dopant_term + self.global_bias
        return log_PDR  # [B, 1]
