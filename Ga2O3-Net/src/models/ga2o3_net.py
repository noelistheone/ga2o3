"""
Ga2O3Net — Full multi-branch fusion model.

Integrates all three stages:
  Stage 1 (frozen): CGCNNEncoder → 64-dim structure embedding
  Stage 2:          CompositionStream (64) + ProcessStream (32)
  Stage 3:          FusionLayer → PredictionHead(s)

Each sample is described by three complementary representations:
  - Structure stream : doped β-Ga₂O₃ crystal graph → GNN → 64-dim
  - Composition stream: DopantSpec string (element + concentration) → XenonPy → MLP → 64-dim
  - Process stream  : [temperature_C, time_min, o2_fraction,
                       is_plasma_enhanced, has_nitrogen_species]  (continuous, normalised)
                     + fabrication method index → Embedding(6, 3) (learned)
                     → MLP → 32-dim
  All three are fused (concat or attention) → 160-dim → two PredictionHeads.

Usage::

    from src.data.experimental_dataset import build_process_tensor

    model = Ga2O3Net.from_pretrained("checkpoints/pretrained_encoder.pt", ...)
    process = build_process_tensor(
        temperature_C=800.0, time_min=90.0,
        atmosphere="O2_plasma", method="RF magnetron sputtering",
    ).unsqueeze(0)  # [1, 10]
    mean, std = model.mc_predict(graph_batch, ["Mg:0.026500"], process)
"""

import os
import logging

import torch
import torch.nn as nn

from src.models.cgcnn_encoder import CGCNNEncoder
from src.models.composition_stream import CompositionStream
from src.models.dopant_stream import DopantStream
from src.models.process_stream import ProcessStream
from src.models.fusion import (
    ConcatFusion,
    SelfAttentionFusion,
    FiLMFusion,
    BiFiLMFusion,
    CrossStreamAttentionFusion,
)
from src.models.moe_head import MoEHead
from src.models.physics_features import (
    compute_physics_features,
    PHYSICS_FEATURE_DIM,
)
from src.models.physics_envelope import (
    PhysicsEnvelopePDR,
    PhysicsEnvelopeVC,
)
from src.models.brouwer_head import BrouwerHeadVC, BrouwerHeadPDR
from src.models.element_hypernet import ElementHyperNet

logger = logging.getLogger(__name__)


class PhysicsStream(nn.Module):
    """
    Phase 24A: physics-feature encoder.

    LayerNorm → 2-layer MLP → out_dim. Inputs are 8 interpretable scalars
    from `physics_features.compute_physics_features`. The LayerNorm
    handles their disparate scales (E_photon ~5 eV vs kT ~0.07 eV) without
    needing a fitted dataset scaler.
    """

    def __init__(self, hidden_dim: int = 32, out_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.in_dim = PHYSICS_FEATURE_DIM
        self.out_dim = out_dim
        self.input_norm = nn.LayerNorm(PHYSICS_FEATURE_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(PHYSICS_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, physics: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.input_norm(physics))


class PredictionHead(nn.Module):
    """
    Lightweight MLP regression head.

    Args:
        in_dim: Input dimension (fused embedding size).
        hidden_dims: List of hidden layer sizes.
        out_dim: Output scalar count (1 per target).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int = 160,
        hidden_dims: list[int] = [64, 32],
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Ga2O3Net(nn.Module):
    """
    Full Ga2O3-Net: Structure + Composition + Process → Fusion → Prediction.

    Args:
        encoder: Pretrained CGCNNEncoder (will be frozen).
        composition_stream: CompositionStream.
        process_stream: ProcessStream.
        fusion_mode: "concat" or "attention".
        target_cols: List of target property names (one head per target).
        head_hidden_dims: Hidden dims for each PredictionHead.
        head_dropout: Dropout for each PredictionHead.
        attention_heads: Num attention heads (only used when fusion_mode="attention").
    """

    STREAM_DIMS = {
        "structure": 64,
        "composition": 64,
        "process": 32,
    }

    def __init__(
        self,
        encoder: CGCNNEncoder,
        composition_stream: CompositionStream,
        process_stream: ProcessStream,
        target_cols: list[str],
        fusion_mode: str = "concat",
        head_hidden_dims: list[int] = [64, 32],
        head_dropout: float = 0.3,
        attention_heads: int = 4,
        dopant_stream: DopantStream | None = None,
        head_type: str = "single",
        num_experts: int = 4,
        moe_routing: str = "soft",
        # Phase 24A — physics feature stream
        physics_features: bool = False,
        physics_features_hidden: int = 32,
        physics_features_out_dim: int = 16,
        physics_features_dropout: float = 0.1,
        # Phase 24B — physics envelopes (applied at output)
        physics_envelope_pdr: bool = False,
        physics_envelope_vc: bool = False,
        # Phase 25-soft: if False, envelope module exists for trainer-side
        # hinge loss but doesn't multiplicatively transform forward output.
        physics_envelope_apply: bool = True,
        # Phase 30B / Path D: anchor BrouwerHeadVC's E_f to the empirical
        # V_O formation-energy proxy (donor/acceptor scaling). Only used when
        # head_type == "brouwer".
        brouwer_anchor_to_proxy: bool = False,
        # Phase 41 V2: per-method scalar offset on log[V_O] in BrouwerHeadVC.
        # Lets multi-method training absorb method-specific baselines without
        # contaminating element-specific dopant_term. Only head_type == "brouwer".
        brouwer_use_method_offset: bool = False,
        brouwer_n_methods: int = 6,
        # Phase 45 V10: per-valence-class scalar offset on log[V_O].
        # 4 classes: 0=acceptor (v=2), 1=isovalent (v=3 / unknown),
        # 2=donor (v=4), 3=super_donor (v≥5).
        brouwer_use_class_offset: bool = False,
        brouwer_n_classes: int = 4,
        # Phase 46 V11: element-descriptor hypernetwork. Replaces the V10
        # class_offset's coarse 4-class scalar with a small MLP that maps
        # 5-dim chemistry descriptors [EN, radius, carrier, Δv, Δperiod] →
        # scalar log_VO offset. Continuous in chemistry space → unseen
        # rare-dopant elements (Ge/Bi/Sb at N=1) inherit physics from
        # labeled neighbors via descriptor interpolation. Only used when
        # head_type == "brouwer".
        # Phase 47 V13: hypernet_final_init controls final-layer weight init.
        # V11 used "zeros" → collapsed; V13 uses "small_random" so hidden
        # layers receive non-zero gradient from epoch 0.
        # Phase 47 V14: hypernet_form controls how H output combines with the
        # Brouwer head. "additive" (V11/V13 default) adds H to log_VO directly;
        # "multiplicative" multiplies the latent dopant_term by (1 + tanh(H))
        # — by construction H cannot collapse to a constant.
        # Phase 47 V15: inject_descriptor concatenates per-row weighted-sum
        # 5-dim descriptor onto fused [B, 160] → [B, 165] before the head,
        # bypassing the HN gating layer entirely (no separate H net).
        brouwer_use_hypernet: bool = False,
        hypernet_hidden: int = 16,
        hypernet_mid: int = 8,
        hypernet_final_init: str = "zeros",
        hypernet_form: str = "additive",
        inject_descriptor: bool = False,
        # Phase 43 V5: distil pretrained encoder's `pretrain_head` outputs
        # (bandgap + formation_energy_per_atom — method-INVARIANT chemistry
        # targets) from the fused embedding. Auxiliary head; trainer-side loss.
        pretrain_distill_aux: bool = False,
        pretrain_distill_dim: int = 2,
        # Phase 53 V53-γ1: concentration-dependent dopant_term in BrouwerHeadVC.
        # When True, latent_net outputs 4 logits and dopant_term =
        #   sign(class) · softplus(a_logit + a_e) · log10(c/c_ref) + (b + b_e)
        # where (a_e, b_e) come from ElementHyperNet (out_dim=2 enforced).
        # Bakes within-DOI monotonicity into the architecture; soft hinge loss
        # weight should be reduced to ~0.5 in fusion.yaml.
        brouwer_use_conc_dep_dopant: bool = False,
        brouwer_conc_ref_at_frac: float = 0.01,
        # Phase 53 V53-ζ: PDR↔V_O latent InfoNCE projection head.
        # When enabled, the trainer reads model.contrastive_proj(fused) and
        # adds an InfoNCE loss with positives = same (DOI, dopant_label).
        # Loss is added in finetune_trainer; this just wires the projection
        # module into the model state_dict so it persists across save/load.
        contrastive_enabled: bool = False,
        contrastive_dim: int = 32,
        contrastive_hidden: int = 64,
        # Phase 53 V53-α: KROGER frozen expert distill aux loss.
        # When `kroger_expert_path` is set, loads EncoderKroger checkpoint and
        # freezes it. Trainer computes target = kroger_expert(process, dopant)
        # and pulls fused→target via aux_distill_kroger_head. Provides
        # method/T/pO2/conc-aware V_O signal even on PDR-only or unlabeled rows.
        kroger_expert_enabled: bool = False,
        kroger_expert_path: str | None = None,
        # Phase 53 V53-η: KKT-Hardnet Brouwer head with full Kröger-Vink charge
        # neutrality (V_O × 3 + V_Ga × 3 + dopant). Replaces BrouwerHeadVC for
        # VC target when enabled. Bisection-based εF solver (differentiable).
        brouwer_use_kkt: bool = False,
        kkt_ef_swing: float = 0.5,
        kkt_n_newton: int = 8,
        # Phase 54 V54-A1: self-distill aux head consuming V53-ζ pseudo-targets
        # on V_O-unlabeled rows. Trainer reads `aux_self_distill_head(fused)`
        # and computes MSE against pseudo_log_vo (standardized) with
        # inverse-variance weighting (lower pseudo_std → higher weight).
        self_distill_enabled: bool = False,
        # Phase 54 V54-A2: SNGP head (spectral norm + RFF Gaussian process)
        # as sibling to BrouwerHeadVC. Provides distance-aware posterior var.
        # Hyperparams from Wei et al. medRxiv 2025.02.26.25322983 §2.3.
        sngp_enabled: bool = False,
        sngp_hidden_dim: int = 256,
        sngp_rff_dim: int = 512,
        sngp_ridge: float = 1e-4,
        sngp_ema: float = 0.999,
        # Phase 54 V54-B1: charge-state V_O / V_Ga frozen experts (sibling
        # to KROGER expert). Distilled via detach()+MSE aux loss only.
        charge_vo_expert_path: str | None = None,
        vga_expert_path: str | None = None,
        # V57-B1-Quartet+1: 4th DFT-scan frozen expert. Same detach() pattern.
        dft_scan_expert_path: str | None = None,
        # V57-MACE-Latent: cached 17×256-d MACE-MP-0 features → 64-d projection,
        # frozen, detached. Used as SINCERE positive pair (trainer reads
        # `model.mace_latent_for(dopant_specs)` and concatenates to fused emb
        # for contrastive loss only — NOT injected into the regression head).
        mace_latent_proj_path: str | None = None,
        mace_latent_dim: int = 64,
        # ── V58 — DFT-anchored stream (M4) + FiLM-208 (M8) + DFTAnchoredBrouwerHead (M9).
        # All gated on `dft_stream_enabled`; when False the model is byte-identical to V5x.
        # Enabling adds z_LLM(32) + φ_DFT(16) to the fused embedding (160→208) so every
        # downstream head (incl. aux contrastive/distill heads) auto-builds at 208.
        # head_type="brouwer_dft" routes VC → DFTAnchoredBrouwerHead. z_LLM is a cached
        # per-(paper,dopant,atm,T) lookup, or zeros when llm_cache_path is None (V58-lite-DFT).
        dft_stream_enabled: bool = False,
        dft_features_cache_path: str | None = None,
        llm_cache_path: str | None = None,
        dft_max_residual: float = 0.3,
        # ── Phase 59 — V59-CALM zero-gated Δ-residual (doc §3.4). EXTENDS the
        # V55-Ext self.fusion(*streams) path; the closed-form Brouwer floor u₀ is
        # UNCHANGED. New modalities (z_expert/φ_DFT/z_MACE) enter ONLY a zero-gated
        # residual Δ added to ŷ_phys (C2 path separation); Δ=0 at init (zero w_out)
        # so the model is byte-identical to V55-Ext at step 0 (no-regression-at-init).
        # ⚠ MUST NOT be combined with dft_stream_enabled (that is the V58 M-route
        # leak: concat into the 208-d magnitude latent). calm_enabled requires the
        # plain `film`/`concat` fusion path.
        calm_enabled: bool = False,
        calm_expert_cache_path: str | None = None,   # z_expert_v59.npz (keys/features 16-d)
        calm_dft_cache_path: str | None = None,       # dft_features_cache_v58.npz (φ_DFT 16-d)
        calm_mace_proj_path: str | None = None,       # mace_proj_64d.pt (z_MACE 64-d) — reuses mace_latent
        calm_fusion_rank: int = 4,
        calm_hidden: int = 16,
        calm_modality_dropout: float = 0.3,
        calm_gate_init: float = 0.0,
        calm_use_expert: bool = True,
        calm_use_dft: bool = True,
        calm_use_mace: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.composition_stream = composition_stream
        self.process_stream = process_stream
        self.dopant_stream = dopant_stream
        self.target_cols = target_cols
        self.fusion_mode = fusion_mode
        self.head_type = head_type
        self.num_experts = num_experts

        struct_dim = encoder.embedding_dim
        comp_dim = composition_stream.out_dim
        proc_dim = process_stream.out_dim
        dopant_dim = dopant_stream.out_dim if dopant_stream is not None else 0

        # Phase 24A: physics stream (8 scalars → MLP → out_dim) inserted
        # before the process stream so FiLM/BiFiLM still see proc at index -1.
        if physics_features:
            self.physics_stream = PhysicsStream(
                hidden_dim=physics_features_hidden,
                out_dim=physics_features_out_dim,
                dropout=physics_features_dropout,
            )
            phys_dim = physics_features_out_dim
        else:
            self.physics_stream = None
            phys_dim = 0

        stream_dims = [struct_dim, comp_dim]
        if dopant_dim:
            stream_dims.append(dopant_dim)
        if phys_dim:
            stream_dims.append(phys_dim)
        stream_dims.append(proc_dim)
        fused_dim = sum(stream_dims)
        self.fused_dim = fused_dim

        # ── V58: append z_LLM (32) + φ_DFT (16) streams so fused_dim becomes 208 and
        # all heads (incl. aux) build at 208. Gated; V5x path untouched when disabled.
        self.dft_stream_enabled = bool(dft_stream_enabled)
        self.dft_stream = None
        self._dft_cache_keys = None
        if self.dft_stream_enabled:
            from src.models.dft_stream import DFTStream
            self.dft_stream = DFTStream(feature_dim=16, out_dim=16)
            self._z_llm_dim = 32
            stream_dims = stream_dims + [self._z_llm_dim, self.dft_stream.out_dim]
            fused_dim = sum(stream_dims)
            self.fused_dim = fused_dim
            self._load_dft_cache(dft_features_cache_path, llm_cache_path)

        if fusion_mode == "concat":
            self.fusion = ConcatFusion()
        elif fusion_mode == "attention":
            self.fusion = SelfAttentionFusion(
                stream_dims=stream_dims,
                token_dim=64,
                num_heads=attention_heads,
                out_dim=fused_dim,
            )
        elif fusion_mode == "film":
            # Phase 13A: element-conditioned process modulation; same output dim
            # as concat (sum of stream dims) so head config can be reused.
            self.fusion = FiLMFusion(
                comp_dim=comp_dim,
                proc_dim=proc_dim,
                film_hidden=64,
                film_dropout=0.1,
            )
        elif fusion_mode == "bifilm":
            # Phase 14A: bidirectional FiLM (comp ↔ proc).
            self.fusion = BiFiLMFusion(
                comp_dim=comp_dim,
                proc_dim=proc_dim,
                film_hidden=64,
                film_dropout=0.1,
            )
        elif fusion_mode == "cross_attn":
            # Phase 13B: 1-layer transformer with [CLS] pooling.
            # CRITICAL: this changes fused_dim to token_dim → must update.
            xattn_token_dim = 64
            self.fusion = CrossStreamAttentionFusion(
                stream_dims=stream_dims,
                token_dim=xattn_token_dim,
                num_heads=attention_heads,
                out_dim=xattn_token_dim,
                dropout=0.2,
            )
            fused_dim = xattn_token_dim
            self.fused_dim = fused_dim
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

        # V58: override fusion with FiLM-208 (M8). It consumes the full 208-d concat
        # (struct, comp, proc, z_LLM, φ_DFT) as a single tensor; get_embedding assembles
        # the concat in the V58 branch. fusion_mode in the config is ignored when enabled.
        if self.dft_stream_enabled:
            from src.models.film_fusion_v58 import FiLMFusion208
            self.fusion = FiLMFusion208(input_dim=self.fused_dim, hidden_dim=128, n_heads=4)

        # Prediction heads: single shared MLP, or MoE soft-routed K experts
        # Accept legacy "mlp" alias for "single"
        if head_type in ("mlp", "MLP"):
            head_type = "single"
            self.head_type = "single"
        if head_type == "moe":
            # Gate input: prefer dopant_stream (chemistry-only, 16-d) if available;
            # otherwise use composition_stream (XenonPy-derived, 64-d).
            gate_in_dim = dopant_dim if dopant_dim > 0 else comp_dim
            self.gate_in_dim = gate_in_dim
            self.heads = nn.ModuleDict({
                col: MoEHead(
                    in_dim=fused_dim,
                    gate_in_dim=gate_in_dim,
                    hidden_dims=head_hidden_dims,
                    out_dim=1,
                    num_experts=num_experts,
                    dropout=head_dropout,
                    routing=moe_routing,
                )
                for col in target_cols
            })
        elif head_type == "single":
            self.heads = nn.ModuleDict({
                col: PredictionHead(fused_dim, head_hidden_dims, 1, head_dropout)
                for col in target_cols
            })
        elif head_type == "brouwer":
            # Phase 30 (VC) + Phase 31 (PDR): per-target closed-form physics heads.
            # `vacancy_concentration` → BrouwerHeadVC (defect equilibrium)
            # `photo_dark_ratio`      → BrouwerHeadPDR (Arrhenius + Ohm)
            # Anything else → fallback PredictionHead.
            # Phase 47 V15: when inject_descriptor=True, only the VC head sees
            # the augmented (fused + 5-dim descriptor) embedding. PDR head and
            # aux_distill_head continue to use fused_dim=160.
            from src.data.element_descriptors import DESCRIPTOR_DIM as _DESC_DIM
            vc_in_dim = fused_dim + _DESC_DIM if bool(inject_descriptor) else fused_dim
            heads = {}
            for col in target_cols:
                if col == "vacancy_concentration":
                    if bool(brouwer_use_kkt):
                        # Phase 53 V53-η: KKT-Hardnet Brouwer (full Kröger-Vink)
                        from src.models.brouwer_head_kkt import BrouwerHeadKKT
                        heads[col] = BrouwerHeadKKT(
                            in_dim=vc_in_dim,
                            hidden_dim=head_hidden_dims[0] if head_hidden_dims else 64,
                            dropout=head_dropout,
                            ef_swing=float(kkt_ef_swing),
                            n_newton=int(kkt_n_newton),
                            use_method_offset=brouwer_use_method_offset,
                            n_methods=brouwer_n_methods,
                        )
                    else:
                        heads[col] = BrouwerHeadVC(
                            in_dim=vc_in_dim,
                            hidden_dim=head_hidden_dims[0] if head_hidden_dims else 64,
                            dropout=head_dropout,
                            anchor_to_proxy=brouwer_anchor_to_proxy,
                            use_method_offset=brouwer_use_method_offset,
                            n_methods=brouwer_n_methods,
                            use_class_offset=brouwer_use_class_offset,
                            n_classes=brouwer_n_classes,
                            use_conc_dep_dopant=brouwer_use_conc_dep_dopant,
                            conc_ref_at_frac=brouwer_conc_ref_at_frac,
                        )
                elif col == "photo_dark_ratio":
                    heads[col] = BrouwerHeadPDR(
                        in_dim=fused_dim,
                        hidden_dim=head_hidden_dims[0] if head_hidden_dims else 64,
                        dropout=head_dropout,
                    )
                else:
                    heads[col] = PredictionHead(fused_dim, head_hidden_dims, 1, head_dropout)
            self.heads = nn.ModuleDict(heads)
        elif head_type == "brouwer_dft":
            # V58: VC → DFTAnchoredBrouwerHead (M9); PDR → BrouwerHeadPDR (V55-Ext form,
            # at fused_dim=208); anything else → PredictionHead.
            from src.models.brouwer_head_dft import DFTAnchoredBrouwerHead
            heads = {}
            for col in target_cols:
                if col == "vacancy_concentration":
                    heads[col] = DFTAnchoredBrouwerHead(
                        fused_dim=fused_dim, max_residual=float(dft_max_residual),
                        fallback_hidden=head_hidden_dims[0] if head_hidden_dims else 32,
                        dropout=head_dropout,
                    )
                elif col == "photo_dark_ratio":
                    heads[col] = BrouwerHeadPDR(
                        in_dim=fused_dim,
                        hidden_dim=head_hidden_dims[0] if head_hidden_dims else 64,
                        dropout=head_dropout,
                    )
                else:
                    heads[col] = PredictionHead(fused_dim, head_hidden_dims, 1, head_dropout)
            self.heads = nn.ModuleDict(heads)
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

        # Phase 24B / 25-soft — physics envelopes.
        # `apply_to_output` controls whether they multiplicatively transform the
        # head output (Phase 24B, retired) or just hold trainable physics
        # constants for the trainer's hinge loss to read (Phase 25-soft).
        self.physics_envelope_pdr = None
        self.physics_envelope_vc = None
        if physics_envelope_pdr:
            if "photo_dark_ratio" not in target_cols:
                raise ValueError("physics_envelope_pdr requires photo_dark_ratio in target_cols")
            self.physics_envelope_pdr = PhysicsEnvelopePDR(
                target_idx=target_cols.index("photo_dark_ratio"),
                apply_to_output=physics_envelope_apply,
            )
        if physics_envelope_vc:
            if "vacancy_concentration" not in target_cols:
                raise ValueError("physics_envelope_vc requires vacancy_concentration in target_cols")
            self.physics_envelope_vc = PhysicsEnvelopeVC(
                target_idx=target_cols.index("vacancy_concentration")
            )

        # Phase 43 V5: auxiliary distillation head from fused → pretrain targets
        # (bandgap + formation_energy_per_atom). Loss applied on ALL rows,
        # regardless of method, since targets are intrinsic chemistry —
        # method-INVARIANT. Pulls non-sputter rows into the chemistry-relevant
        # subspace of fused latent without polluting V_O / Brouwer scope.
        self.pretrain_distill_aux = bool(pretrain_distill_aux)
        if self.pretrain_distill_aux:
            self.aux_distill_head = nn.Linear(fused_dim, int(pretrain_distill_dim))
        else:
            self.aux_distill_head = None

        # Phase 46 V11: element-descriptor hypernetwork. Only valid for
        # head_type=="brouwer" (since the offset is added to BrouwerHeadVC's
        # log_VO output, not to a generic regression head).
        self.brouwer_use_hypernet = bool(brouwer_use_hypernet)
        self.hypernet_form = str(hypernet_form)
        if self.hypernet_form not in ("additive", "multiplicative"):
            raise ValueError(
                f"hypernet_form must be 'additive' or 'multiplicative', got {self.hypernet_form!r}"
            )
        # Phase 53 V53-γ1: when conc_dep_dopant is on, ElementHyperNet outputs
        # 2 scalars (a_e, b_e); otherwise default 1 scalar (V52b log_VO offset).
        self.brouwer_use_conc_dep_dopant = bool(brouwer_use_conc_dep_dopant)
        self.brouwer_conc_ref_at_frac = float(brouwer_conc_ref_at_frac)
        if self.brouwer_use_conc_dep_dopant and head_type != "brouwer":
            raise ValueError(
                "brouwer_use_conc_dep_dopant=True requires head_type='brouwer'."
            )
        if self.brouwer_use_hypernet:
            if head_type != "brouwer":
                raise ValueError(
                    "brouwer_use_hypernet=True requires head_type='brouwer'."
                )
            from src.data.element_descriptors import DESCRIPTOR_DIM
            hn_out_dim = 2 if self.brouwer_use_conc_dep_dopant else 1
            self.element_hypernet = ElementHyperNet(
                desc_dim=DESCRIPTOR_DIM,
                hidden=int(hypernet_hidden),
                mid=int(hypernet_mid),
                final_init=str(hypernet_final_init),
                out_dim=hn_out_dim,
            )
        else:
            self.element_hypernet = None

        # Phase 47 V15: direct descriptor injection (no HN gating layer).
        # When True, _heads_from_fused concatenates the per-row weighted-sum
        # 5-dim descriptor onto fused before passing to BrouwerHeadVC.
        # Critical: aux_distill_head still uses original fused_dim (descriptor
        # is injected only for the VC head, not the auxiliary distill loss).
        self.inject_descriptor = bool(inject_descriptor)
        if self.inject_descriptor and head_type != "brouwer":
            raise ValueError(
                "inject_descriptor=True requires head_type='brouwer' (only "
                "BrouwerHeadVC consumes the augmented fused embedding)."
            )

        # Phase 53 V53-ζ: optional InfoNCE projection head on fused embedding.
        # Trainer-side loss reads model.contrastive_proj(fused).
        self.contrastive_enabled = bool(contrastive_enabled)
        if self.contrastive_enabled:
            from src.models.contrastive_head import InfoNCEProjector
            self.contrastive_proj = InfoNCEProjector(
                in_dim=fused_dim,
                hidden=int(contrastive_hidden),
                out_dim=int(contrastive_dim),
            )
        else:
            self.contrastive_proj = None

        # Phase 53 V53-α: KROGER frozen expert + distill aux head.
        # encoder_kroger predicts log10[V_O] for given (T, p_O2, dopant, log_c)
        # using analytical β-Ga2O3 Brouwer math. Frozen expert; aux loss pulls
        # fused embedding toward kroger's log_VO prediction.
        self.kroger_expert_enabled = bool(kroger_expert_enabled)
        if self.kroger_expert_enabled:
            if kroger_expert_path is None:
                raise ValueError("kroger_expert_enabled=True requires kroger_expert_path")
            from src.data.kroger_synthetic import ELEMENTS as _KROGER_ELEMS
            from src.models.kroger_expert import load_encoder_kroger
            self.kroger_expert = load_encoder_kroger(kroger_expert_path)
            for p in self.kroger_expert.parameters():
                p.requires_grad_(False)
            self.kroger_expert.eval()
            self.kroger_elements = list(_KROGER_ELEMS)
            self.aux_distill_kroger_head = nn.Linear(fused_dim, 1)
        else:
            self.kroger_expert = None
            self.kroger_elements = None
            self.aux_distill_kroger_head = None

        # Phase 54 V54-A1: self-distill aux head consuming V53-ζ pseudo-targets.
        self.self_distill_enabled = bool(self_distill_enabled)
        if self.self_distill_enabled:
            self.aux_self_distill_head = nn.Linear(fused_dim, 1)
        else:
            self.aux_self_distill_head = None

        # Phase 54 V54-A2: SNGP head as sibling to BrouwerHeadVC.
        # Provides distance-aware posterior variance (alt to MC-dropout) on
        # the V_O residual. Wired only when sngp_enabled=True via physics
        # config. The trainer calls model.reset_sngp_precision() at each
        # fold start and reads (mean, var) at eval time.
        self.sngp_enabled = bool(sngp_enabled)
        if self.sngp_enabled:
            from src.models.sngp_head import SNGPHead
            self.sngp_head = SNGPHead(
                in_dim=fused_dim,
                hidden_dim=int(sngp_hidden_dim),
                rff_dim=int(sngp_rff_dim),
                ridge=float(sngp_ridge),
                momentum=float(sngp_ema),
            )
        else:
            self.sngp_head = None

        # Phase 54 V54-B1: charge-state V_O frozen expert + aux distill head.
        if charge_vo_expert_path is not None and os.path.exists(charge_vo_expert_path):
            from src.models.charge_vo_expert import load_charge_vo_expert
            self.charge_vo_expert = load_charge_vo_expert(charge_vo_expert_path)
            for p in self.charge_vo_expert.parameters():
                p.requires_grad_(False)
            self.charge_vo_expert.eval()
            self.aux_distill_charge_vo_head = nn.Linear(fused_dim, 3)  # 3 charges
        else:
            self.charge_vo_expert = None
            self.aux_distill_charge_vo_head = None

        # Phase 54 V54-B1: V_Ga frozen expert + aux distill head.
        if vga_expert_path is not None and os.path.exists(vga_expert_path):
            from src.models.vga_expert import load_vga_expert
            self.vga_expert = load_vga_expert(vga_expert_path)
            for p in self.vga_expert.parameters():
                p.requires_grad_(False)
            self.vga_expert.eval()
            self.aux_distill_vga_head = nn.Linear(fused_dim, 3)
        else:
            self.vga_expert = None
            self.aux_distill_vga_head = None

        # V57-B1-Quartet+1: 4th DFT-scan frozen expert (3-charge head, predicts
        # log10[V_O^q=0,+1,+2] from element + log_c + log_T + log_pO2 + q-onehot).
        # Trainer reads `model.dft_scan_predict(dopant_specs, process)` and adds
        # MSE distill on `aux_distill_dft_scan_head(fused)` with λ_aux=0.05.
        if dft_scan_expert_path is not None and os.path.exists(dft_scan_expert_path):
            from src.models.frozen_encoder_dft_scan import load_frozen_dft_scan
            self.dft_scan_expert = load_frozen_dft_scan(dft_scan_expert_path)
            for p in self.dft_scan_expert.parameters():
                p.requires_grad_(False)
            self.dft_scan_expert.eval()
            self.aux_distill_dft_scan_head = nn.Linear(fused_dim, 3)  # 3 charges
        else:
            self.dft_scan_expert = None
            self.aux_distill_dft_scan_head = None

        # V57-MACE-Latent: 17×256-d cached MACE-MP-0 → 64-d frozen projection.
        # Trainer uses `model.mace_latent_for(dopant_specs)` to fetch a [B, 64]
        # detached tensor for SINCERE InfoNCE positive-pair construction. The
        # projection state-dict is loaded ONCE at module init and never trained
        # (Rule 1 compliance — Phase 18b lesson).
        if mace_latent_proj_path is not None and os.path.exists(mace_latent_proj_path):
            import torch as _torch
            ck = _torch.load(mace_latent_proj_path, map_location="cpu", weights_only=False)
            self._mace_elements: list[str] = list(ck["elements"])
            self._mace_features = ck["mace_mean_dop"]
            self.mace_latent_proj = nn.Linear(int(ck["dim_in"]),
                                              int(ck["dim_out"]), bias=False)
            self.mace_latent_proj.weight.data.copy_(ck["proj_weight"])
            for p in self.mace_latent_proj.parameters():
                p.requires_grad_(False)
            self.mace_latent_proj.eval()
            self.mace_latent_dim = int(ck["dim_out"])
        else:
            self._mace_elements = []
            self._mace_features = None
            self.mace_latent_proj = None
            self.mace_latent_dim = mace_latent_dim

        # ── Phase 59 — V59-CALM zero-gated Δ-residual (doc §3.4) ──────────────
        self.calm_enabled = bool(calm_enabled)
        self.calm_residual = None
        self._calm_expert_keys = None
        self._calm_dft_keys = None
        if self.calm_enabled:
            if self.dft_stream_enabled:
                raise ValueError(
                    "V59-CALM (calm_enabled) is incompatible with dft_stream_enabled "
                    "(the V58 M-route concat leak). Use the plain film/concat fusion path."
                )
            from src.models.calm_residual import CALMResidual
            self._calm_use_mace = bool(calm_use_mace)
            # Load z_expert + φ_DFT caches into buffers (keyed '{cation}|{atm}|{T_bin}').
            expert_dim = self._load_calm_caches(calm_expert_cache_path, calm_dft_cache_path)
            # z_MACE reuses the V57-MACE-Latent projection (loaded above) if requested.
            if calm_use_mace and self.mace_latent_proj is None and calm_mace_proj_path:
                import torch as _torch
                ck = _torch.load(calm_mace_proj_path, map_location="cpu", weights_only=False)
                self._mace_elements = list(ck["elements"])
                self._mace_features = ck["mace_mean_dop"]
                self.mace_latent_proj = nn.Linear(int(ck["dim_in"]), int(ck["dim_out"]), bias=False)
                self.mace_latent_proj.weight.data.copy_(ck["proj_weight"])
                for p in self.mace_latent_proj.parameters():
                    p.requires_grad_(False)
                self.mace_latent_proj.eval()
                self.mace_latent_dim = int(ck["dim_out"])
            mace_dim = self.mace_latent_dim if (calm_use_mace and self.mace_latent_proj is not None) else 0
            use_mace_eff = bool(calm_use_mace and self.mace_latent_proj is not None)
            self.calm_residual = CALMResidual(
                u0_dim=self.fused_dim,
                expert_dim=expert_dim if expert_dim else 16,
                dft_dim=16,
                mace_dim=mace_dim if mace_dim else 64,
                fusion_rank=int(calm_fusion_rank),
                hidden=int(calm_hidden),
                dropout=0.0,
                modality_dropout=float(calm_modality_dropout),
                gate_init=float(calm_gate_init),
                use_expert=bool(calm_use_expert and self._calm_expert_keys is not None),
                use_dft=bool(calm_use_dft and self._calm_dft_keys is not None),
                use_mace=use_mace_eff,
            )

    def dft_scan_predict(
        self, dopant_specs: list[str], process: torch.Tensor
    ) -> torch.Tensor:
        """V57-Quartet+1: frozen 4th expert returns [B, 3] log10[V_O^q] for q=0,+1,+2.

        Returns a detached tensor — the caller is responsible for distillation
        loss. If the expert isn't loaded, returns a zero tensor of the right
        shape so the trainer's `MSE(aux_head, dft_scan_pred)` becomes a no-op.
        """
        B = len(dopant_specs)
        device = process.device
        if self.dft_scan_expert is None:
            return torch.zeros(B, 3, device=device)
        # Build element one-hot + continuous features matching the expert's
        # training schema (see scripts/35_v57_quartet_plus_pretrain.py:df_to_tensors).
        from src.data.dopant_spec import DopantSpec
        from src.data.kroger_synthetic import ELEMENTS, C_REF
        elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}
        elem_idx_list, c_list = [], []
        for s in dopant_specs:
            try:
                ds = DopantSpec.parse(s)
                if ds.is_undoped or not ds.components:
                    elem_idx_list.append(0); c_list.append(float(C_REF))
                else:
                    cat = max(ds.components, key=lambda c: c.conc)
                    elem_idx_list.append(elem_to_idx.get(cat.cation, 0))
                    c_list.append(max(float(cat.conc), 1e-6))
            except Exception:
                elem_idx_list.append(0); c_list.append(float(C_REF))
        elem_idx = torch.tensor(elem_idx_list, dtype=torch.long, device=device)
        elem_oh = torch.nn.functional.one_hot(elem_idx, len(ELEMENTS)).float()
        log_c = torch.log10(torch.tensor(c_list, dtype=torch.float32,
                                          device=device).clamp(min=1e-6))
        T_C = process[:, 0]
        T_K = (T_C + 273.15).clamp(min=100.0)
        log_T = torch.log10(T_K)
        o2 = process[:, 2].clamp(min=1e-4, max=1.0)
        log_pO2 = torch.log10(o2)
        preds = []
        with torch.no_grad():
            for q in range(3):
                q_oh = torch.nn.functional.one_hot(
                    torch.full((B,), q, dtype=torch.long, device=device),
                    num_classes=3,
                ).float()
                preds.append(self.dft_scan_expert(
                    elem_oh, log_c, log_T, log_pO2, q_oh,
                ))
        return torch.stack(preds, dim=-1).detach()  # [B, 3]

    def mace_latent_for(self, dopant_specs: list[str]) -> torch.Tensor | None:
        """V57-MACE-Latent: return frozen 64-d projection for each sample's
        primary dopant. None if projection / features not loaded.

        Output is detached so it can be safely concatenated to fused
        embeddings inside the SINCERE InfoNCE term without gradient leak.
        """
        if self.mace_latent_proj is None or self._mace_features is None:
            return None
        from src.data.dopant_spec import DopantSpec
        device = next(self.mace_latent_proj.parameters()).device
        idx_list = []
        for s in dopant_specs:
            try:
                ds = DopantSpec.parse(s)
                if ds.is_undoped or not ds.components:
                    idx_list.append(-1); continue
                cation = max(ds.components, key=lambda c: c.conc).cation
                idx_list.append(self._mace_elements.index(cation)
                                 if cation in self._mace_elements else -1)
            except Exception:
                idx_list.append(-1)
        # Map -1 → all-zero feature (treated as "no MACE evidence")
        feats = torch.zeros(len(idx_list), int(self.mace_latent_proj.in_features),
                            device=device)
        mace_t = self._mace_features.to(device)
        for i, idx in enumerate(idx_list):
            if idx >= 0:
                feats[i] = mace_t[idx]
        with torch.no_grad():
            z = self.mace_latent_proj(feats)
            z = torch.nn.functional.normalize(z, dim=-1)
        return z.detach()

    # ── Phase 59 — V59-CALM cache loaders + Δ-residual ──────────────────────────
    def _load_calm_caches(self, expert_path, dft_path) -> int:
        """Load z_expert (16-d) + φ_DFT (16-d) caches keyed '{cation}|{atm}|{T_bin}'
        as non-persistent buffers. Returns the z_expert feature dim (0 if absent)."""
        import numpy as np
        expert_dim = 0
        if expert_path is not None and os.path.exists(expert_path):
            de = np.load(expert_path, allow_pickle=True)
            self._calm_expert_keys = {str(k): i for i, k in enumerate(de["keys"])}
            self.register_buffer("_calm_expert_feats",
                                 torch.tensor(de["features"], dtype=torch.float32),
                                 persistent=False)
            expert_dim = int(self._calm_expert_feats.shape[1])
        if dft_path is not None and os.path.exists(dft_path):
            dd = np.load(dft_path, allow_pickle=False)
            self._calm_dft_keys = {str(k): i for i, k in enumerate(dd["keys"])}
            self.register_buffer("_calm_dft_feats",
                                 torch.tensor(dd["features"], dtype=torch.float32),
                                 persistent=False)
        return expert_dim

    def _lookup_calm_expert(self, dopant_specs, process) -> torch.Tensor | None:
        """[B, expert_dim] cached z_expert; NaN rows for cache misses."""
        if self._calm_expert_keys is None:
            return None
        B = len(dopant_specs)
        D = int(self._calm_expert_feats.shape[1])
        out = torch.full((B, D), float("nan"), device=process.device, dtype=torch.float32)
        for i, s in enumerate(dopant_specs):
            key = self._v58_dopant_atm_T_key(s, process[i])
            if key is not None and key in self._calm_expert_keys:
                out[i] = self._calm_expert_feats[self._calm_expert_keys[key]].to(process.device)
        return out

    def _lookup_calm_dft(self, dopant_specs, process) -> torch.Tensor | None:
        """[B, 16] cached φ_DFT; NaN rows for cache misses."""
        if self._calm_dft_keys is None:
            return None
        B = len(dopant_specs)
        D = int(self._calm_dft_feats.shape[1])
        out = torch.full((B, D), float("nan"), device=process.device, dtype=torch.float32)
        for i, s in enumerate(dopant_specs):
            key = self._v58_dopant_atm_T_key(s, process[i])
            if key is not None and key in self._calm_dft_keys:
                out[i] = self._calm_dft_feats[self._calm_dft_keys[key]].to(process.device)
        return out

    def calm_delta(self, fused: torch.Tensor, dopant_specs: list[str],
                   process: torch.Tensor) -> torch.Tensor:
        """V59-CALM Δ correction [B, 1] added to ŷ_phys. Reads frozen detached
        modalities (z_expert/φ_DFT/z_MACE) and the log-conc value process[:, 11].

        NOTE (errata vs design §4.1/§4.2): the DCC loss does NOT take ∂Δ/∂log c via
        autograd — ProcessStream.normalize() routes the StandardScaler through numpy,
        which blocks autograd through process[:, 11]. Instead the trainer
        (`_dcc_loss` in finetune_trainer.py) computes the concentration slope by FINITE
        DIFFERENCES (pred[k+1]−pred[k]) over separately rebuilt concentration grid
        points — the exact discrete twin of scripts/eval_counterfactual_sweep.py
        (train/eval symmetry). Gradients still reach the params through each rebuilt
        prediction. Functionally validated (scripts/61_v59_smoke_test.py)."""
        z_expert = self._lookup_calm_expert(dopant_specs, process) if self.calm_residual.use_expert else None
        phi_dft = self._lookup_calm_dft(dopant_specs, process) if self.calm_residual.use_dft else None
        z_mace = self.mace_latent_for(dopant_specs) if (self.calm_residual.use_mace) else None
        logc = process[:, 11:12]
        return self.calm_residual(fused, z_expert, phi_dft, z_mace, logc)

    def reset_sngp_precision(self) -> None:
        """V54-A2: zero the SNGP RFF precision matrix at fold start so each
        fold accumulates Σ from scratch (no cross-fold leakage)."""
        if getattr(self, "sngp_head", None) is not None:
            self.sngp_head.reset_precision()

    def _maybe_apply_physics_envelope(
        self,
        out: torch.Tensor,
        dopant_specs: list[str],
        process: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PDR / VC physics envelopes if configured."""
        if self.physics_envelope_pdr is None and self.physics_envelope_vc is None:
            return out
        physics = compute_physics_features(dopant_specs, process)
        if self.physics_envelope_pdr is not None:
            out = self.physics_envelope_pdr(out, process, physics)
        if self.physics_envelope_vc is not None:
            out = self.physics_envelope_vc(out, process, physics)
        return out

    def forward(
        self,
        graph,
        dopant_specs: list[str],
        process: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            graph       : PyG Batch from experimental_dataset.collate_fn.
            dopant_specs: list[str] of DopantSpec strings, length B.
                          e.g. ["Fe:0.0265", "SnO2:0.013,MgO:0.013", …]
                          Encodes both dopant element identity and concentration.
            process     : Tensor [B, 10] — raw process condition values.
                          Columns (in order):
                            0  temperature_C        annealing temperature (°C)
                            1  time_min             annealing duration (min)
                            2  o2_fraction          O₂ mole fraction [0, 1]
                            3  is_plasma_enhanced   1.0 if plasma-activated atmosphere
                            4  has_nitrogen_species 1.0 if N₂/N₂O/NH₃ present
                            5  method_sputtering    1.0 if RF/DC magnetron sputtering
                            6  method_pld           1.0 if PLD / laser deposition
                            7  method_cvd_ald       1.0 if CVD / ALD / MOCVD / MBE
                            8  method_wet           1.0 if hydrothermal / sol-gel etc.
                            9  method_evaporation   1.0 if thermal / e-beam evaporation
                          Use build_process_tensor() from experimental_dataset to
                          construct this from human-readable strings.

        Returns:
            Tensor [B, num_targets]
        """
        fused = self.get_embedding(graph, dopant_specs, process)
        if self.head_type == "moe":
            gate_feats = self._get_gate_features(dopant_specs)
            return self._heads_from_fused(
                fused, gate_features=gate_feats,
                dopant_specs=dopant_specs, process=process,
            )
        return self._heads_from_fused(
            fused, dopant_specs=dopant_specs, process=process,
        )

    def get_embedding(
        self,
        graph,
        dopant_specs: list[str],
        process: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return fused embedding without prediction heads.

        Args:
            graph       : PyG Batch.
            dopant_specs: list[str] of DopantSpec strings, length B.
            process     : Tensor [B, 10] — same layout as forward().

        Returns:
            Tensor [B, fused_dim]  (default 160)

        Note: No torch.no_grad() wrapper around the encoder — when the encoder
        is partially unfrozen (unfreeze_last_n), gradients must flow through the
        unfrozen layers. Frozen params (requires_grad=False) are correctly skipped
        by the autograd engine. When the encoder is fully frozen, the behaviour
        is functionally identical to wrapping in torch.no_grad().
        """
        # Structure stream — gradient flows only to unfrozen encoder layers.
        # structure_scale=0.0 ablates the CGCNN contribution (useful when
        # bulk-crystal pretrained features are not informative for the target).
        struct_emb = self.encoder.get_embedding(graph)          # [B, 64]
        scale = getattr(self, "structure_scale", 1.0)
        if scale != 1.0:
            struct_emb = struct_emb * scale

        # Composition stream: arbitrary dopant specs → 64-dim
        comp_emb = self.composition_stream(dopant_specs)        # [B, 64]

        # Process stream: normalises internally
        proc_emb = self.process_stream(process)                 # [B, 32]

        # Stream order: (struct, comp, [dopant], [physics], proc) — physics
        # inserted before proc so FiLM/BiFiLM still see proc at index -1.
        streams: list[torch.Tensor] = [struct_emb, comp_emb]
        if self.dopant_stream is not None:
            streams.append(self.dopant_stream(dopant_specs))
        if self.physics_stream is not None:
            phys_feats = compute_physics_features(dopant_specs, process)
            streams.append(self.physics_stream(phys_feats))
        streams.append(proc_emb)
        if self.dft_stream_enabled:
            # V58 fusion order (doc §3.8.1): [struct, comp, proc, z_LLM, φ_DFT] → 208-d
            dft_features = self._lookup_dft_features(dopant_specs, process)   # [B,16]
            phi_dft, _ = self.dft_stream(dft_features)                        # [B,16]
            z_llm = self._lookup_z_llm(dopant_specs, process)                 # [B,32]
            concat = torch.cat([struct_emb, comp_emb, proc_emb, z_llm, phi_dft], dim=-1)
            return self.fusion(concat)                                       # [B,208]
        return self.fusion(*streams)                             # [B, fused_dim]

    # ── V58 DFT/LLM cache helpers ─────────────────────────────────────────────
    _V58_ATM_O2FRAC = {"Ar": 0.0, "Ar_O2_4_1": 0.2, "Ar_O2_1_1": 0.5, "O2": 1.0}
    _V58_T_BINS = [500, 700, 900, 1100]

    def _load_dft_cache(self, dft_path, llm_path) -> None:
        """Load dft_features_cache (keys→feats) and optional z_llm cache. feats stored
        as a buffer so it follows .to(device). z_llm None → z_LLM=zeros (V58-lite-DFT)."""
        import numpy as np
        if dft_path is None:
            raise ValueError("dft_stream_enabled=True requires dft_features_cache_path")
        d = np.load(dft_path, allow_pickle=False)
        self._dft_cache_keys = {str(k): i for i, k in enumerate(d["keys"])}
        self.register_buffer("_dft_cache_feats",
                             torch.tensor(d["features"], dtype=torch.float32),
                             persistent=False)
        if llm_path is not None:
            dl = np.load(llm_path, allow_pickle=False)
            self._z_llm_cache_keys = {str(k): i for i, k in enumerate(dl["keys"])}
            self.register_buffer("_z_llm_cache_feats",
                                 torch.tensor(dl["features"], dtype=torch.float32),
                                 persistent=False)
        else:
            self._z_llm_cache_keys = None

    def _v58_dopant_atm_T_key(self, spec_str: str, proc_row: torch.Tensor) -> str | None:
        """Build the cache key '{dopant}|{atm}|{T_C}' from a dopant spec + process row."""
        from src.data.dopant_spec import DopantSpec
        try:
            spec = DopantSpec.parse(spec_str.strip())
        except Exception:
            spec = None
        if spec is None or getattr(spec, "is_undoped", False) or not spec.components:
            return None
        # primary cation = highest concentration component
        cation = max(spec.components, key=lambda c: float(c.conc)).cation
        o2 = float(proc_row[2].item())
        atm = min(self._V58_ATM_O2FRAC, key=lambda a: abs(self._V58_ATM_O2FRAC[a] - o2))
        T_C = float(proc_row[0].item())
        T_bin = min(self._V58_T_BINS, key=lambda t: abs(t - T_C))
        return f"{cation}|{atm}|{T_bin}"

    def _lookup_dft_features(self, dopant_specs, process) -> torch.Tensor:
        """[B,16] cached DFT features; NaN rows for dopants absent from the cache
        (M4 imputes φ_DFT; M9 falls back to learned ΔE_f)."""
        B = len(dopant_specs)
        feats = torch.full((B, 16), float("nan"), device=process.device, dtype=torch.float32)
        for i, s in enumerate(dopant_specs):
            key = self._v58_dopant_atm_T_key(s, process[i])
            if key is not None and key in self._dft_cache_keys:
                feats[i] = self._dft_cache_feats[self._dft_cache_keys[key]].to(process.device)
        return feats

    def _lookup_z_llm(self, dopant_specs, process) -> torch.Tensor:
        """[B,32] cached z_LLM, or zeros when no LLM cache (V58-lite-DFT)."""
        B = len(dopant_specs)
        if self._z_llm_cache_keys is None:
            return torch.zeros(B, self._z_llm_dim, device=process.device, dtype=torch.float32)
        z = torch.zeros(B, self._z_llm_dim, device=process.device, dtype=torch.float32)
        for i, s in enumerate(dopant_specs):
            key = self._v58_dopant_atm_T_key(s, process[i])
            if key is not None and key in self._z_llm_cache_keys:
                z[i] = self._z_llm_cache_feats[self._z_llm_cache_keys[key]].to(process.device)
        return z

    def set_dft_target_stats(self, mean: float, std: float) -> None:
        """Trainer hook: propagate per-fold V_O target standardization to the M9 head
        so its absolute DFT anchor maps into standardized space. No-op for non-V58 heads."""
        if "vacancy_concentration" in self.heads:
            h = self.heads["vacancy_concentration"]
            if hasattr(h, "set_target_stats"):
                h.set_target_stats(mean, std)

    # ── Phase 45 V10 helper ──────────────────────────────────────────────────
    # Per-valence-class lookup for the BrouwerHeadVC's optional class_offset.
    # Class enumeration (matches scripts/eval_physics_diag_table.py):
    #   0 = acceptor      (v=2: Mg, Zn, Cu)
    #   1 = isovalent     (v=3 or unknown: Al, Fe, B, V, Er, Eu, undoped)
    #   2 = donor         (v=4: Si, Sn, Ti, Ge)
    #   3 = super_donor   (v≥5: Sb, Ta, Bi, W)
    _ELEM_CLASS_MAP = {
        # acceptors
        "Mg": 0, "Zn": 0, "Cu": 0,
        # isovalent / d-block trivalent / unknown
        # V52 (2026-05-06): Sb³⁺/Bi³⁺ moved here. Both have ns² lone pair
        # (Bi 6s², Sb 5s²) and prefer +3 substitution on Ga³⁺ site, not +5.
        # Class-aware monotonicity loss should treat them as isovalent
        # (no expected sign of dV_O/d[dopant]) rather than donor-like.
        # Physics: J. Phys. Chem. C 2025, 10.1021/acs.jpcc.5c02687 (Bi
        # raises VBM via lone-pair, doesn't donate electrons).
        "Al": 1, "Fe": 1, "B": 1, "V": 1, "Er": 1, "Eu": 1, "F": 1,
        "Sb": 1, "Bi": 1,    # ← V52: was 3 (super-donor)
        # donors (true classical donors with no active lone pair)
        "Si": 2, "Sn": 2, "Ti": 2, "Ge": 2,
        # super-donors (no lone pair — d-block / 4f)
        "Ta": 3, "W": 3,     # ← Bi/Sb removed
    }

    def _dopant_specs_to_class_idx(self, dopant_specs: list[str],
                                    device: torch.device) -> torch.Tensor:
        """Parse dopant_specs into a [B] long tensor of valence-class indices."""
        from src.data.dopant_spec import DopantSpec
        out = []
        for s in dopant_specs:
            try:
                spec = DopantSpec.parse(s)
                if spec.is_undoped or not spec.components:
                    out.append(1)  # default isovalent
                    continue
                cation = max(spec.components, key=lambda c: c.conc).cation
                out.append(self._ELEM_CLASS_MAP.get(cation, 1))
            except Exception:
                out.append(1)
        return torch.tensor(out, dtype=torch.long, device=device)

    def _build_hypernet_inputs(self, dopant_specs: list[str],
                                device: torch.device,
                                return_total_conc: bool = False):
        """Phase 46 V11: build [B, K_max, DESCRIPTOR_DIM] descriptors + [B, K_max]
        *normalized* fractions for the ElementHyperNet. Undoped rows route to
        the "undoped" descriptor with weight 1.0.

        Phase 53 V53-γ1: when ``return_total_conc=True`` also returns
        [B, 1] tensor of total dopant atomic fraction (sum of components,
        before normalization). Undoped rows have total_conc set to ``conc_ref``
        of the head, so log10(c/c_ref) = 0 (no concentration response).
        """
        from src.data.dopant_spec import DopantSpec
        from src.data.element_descriptors import ELEMENT_DESCRIPTORS, DESCRIPTOR_DIM

        parsed: list[list[tuple[str, float]]] = []
        is_undoped: list[bool] = []
        K_max = 1
        for s in dopant_specs:
            try:
                spec = DopantSpec.parse(s.strip())
            except Exception:
                spec = None
            if spec is None or spec.is_undoped or not spec.components:
                parsed.append([("undoped", 1.0)])
                is_undoped.append(True)
                continue
            items = [(c.cation, max(float(c.conc), 0.0)) for c in spec.components]
            # Drop zero-concentration entries (otherwise the normalized fraction
            # divides by 0 if all concs are 0; spec.is_undoped already handles
            # the truly-undoped case).
            items = [it for it in items if it[1] > 0.0] or [("other", 1.0)]
            K_max = max(K_max, len(items))
            parsed.append(items)
            is_undoped.append(False)

        B = len(dopant_specs)
        descriptors = torch.zeros(B, K_max, DESCRIPTOR_DIM,
                                   device=device, dtype=torch.float32)
        weights = torch.zeros(B, K_max, device=device, dtype=torch.float32)
        total_conc = torch.zeros(B, 1, device=device, dtype=torch.float32)
        fallback_desc = ELEMENT_DESCRIPTORS["other"]
        for i, items in enumerate(parsed):
            total = sum(c for _, c in items)
            if total <= 0:
                total = 1.0
            for k, (cation, conc) in enumerate(items):
                d = ELEMENT_DESCRIPTORS.get(cation, fallback_desc)
                descriptors[i, k, :] = torch.tensor(d, device=device, dtype=torch.float32)
                weights[i, k] = conc / total
            # V53-γ1: undoped rows → set total_conc to c_ref so log_c=0.
            # Otherwise use the absolute sum of dopant atomic fractions.
            if is_undoped[i]:
                total_conc[i, 0] = float(self.brouwer_conc_ref_at_frac)
            else:
                total_conc[i, 0] = float(total)
        if return_total_conc:
            return descriptors, weights, total_conc
        return descriptors, weights

    def _heads_from_fused(
        self,
        fused: torch.Tensor,
        gate_features: torch.Tensor | None = None,
        dopant_specs: list[str] | None = None,
        process: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run prediction heads on a fused embedding.

        For MoE heads, gate features must come from somewhere:
          - Pass `gate_features` directly (precomputed), OR
          - Pass `dopant_specs` and we'll call `_get_gate_features` internally.
        Single heads ignore both arguments.

        Phase 24B: when ``process`` is supplied AND the model has physics
        envelopes wired, the envelope is applied to the head output. The trainer
        must pass ``process`` so envelope params get gradient signal — without
        it the envelope is silently skipped (init-only at deployment).
        """
        if self.head_type == "moe":
            if gate_features is None:
                if dopant_specs is None:
                    # Last-resort fallback: try to derive from a cached batch on the model.
                    # Trainers should always pass one of the two; raise to surface bugs.
                    raise ValueError(
                        "MoE heads need either gate_features or dopant_specs; "
                        "none provided.")
                gate_features = self._get_gate_features(dopant_specs)
            preds = [self.heads[col](fused, gate_features) for col in self.target_cols]
        elif self.head_type == "brouwer":
            if process is None:
                raise ValueError(
                    "Brouwer head requires `process` tensor in _heads_from_fused; "
                    "trainer/eval path must pass it.")
            # Phase 41 V2: extract method index for BrouwerHeadVC's optional
            # per-method offset. Same convention as ProcessStream: bits at
            # [continuous_dim : continuous_dim+5] are sputter/PLD/CVD-ALD/wet/evap;
            # all-zero → "other" (index 5). Reuse the process_stream's method
            # to stay layout-safe across v2 / v3 process tensors.
            method_idx = None
            class_idx = None
            element_offset = None
            dopant_term_multiplier = None
            dopant_total_conc = None
            dopant_a_offset = None
            dopant_b_offset = None
            fused_for_vc = fused
            for col in self.target_cols:
                if col == "vacancy_concentration":
                    h = self.heads[col]
                    if isinstance(h, BrouwerHeadVC) and getattr(h, "use_method_offset", False):
                        method_idx = self.process_stream._method_bits_to_idx(process)
                    # V53-γ1 always needs class_idx (sign LUT) and total_conc.
                    needs_conc_dep = isinstance(h, BrouwerHeadVC) and \
                        getattr(h, "use_conc_dep_dopant", False)
                    if (isinstance(h, BrouwerHeadVC) and
                        (getattr(h, "use_class_offset", False) or needs_conc_dep)) \
                       and dopant_specs is not None:
                        class_idx = self._dopant_specs_to_class_idx(dopant_specs, fused.device)
                    if isinstance(h, BrouwerHeadVC) and self.element_hypernet is not None \
                       and dopant_specs is not None:
                        if needs_conc_dep:
                            descriptors, weights, dopant_total_conc = \
                                self._build_hypernet_inputs(
                                    dopant_specs, fused.device, return_total_conc=True)
                        else:
                            descriptors, weights = self._build_hypernet_inputs(
                                dopant_specs, fused.device)
                        h_out = self.element_hypernet(descriptors, weights)
                        if needs_conc_dep:
                            # V53-γ1: HN output is (a_e, b_e) per row; route into
                            # dopant_a_offset / dopant_b_offset of BrouwerHeadVC.
                            dopant_a_offset = h_out[:, 0:1]
                            dopant_b_offset = h_out[:, 1:2]
                        # Phase 47 V14: route H output as multiplicative
                        # modulation of dopant_term, instead of additive on
                        # log_VO. Cannot collapse to constant by construction.
                        elif self.hypernet_form == "multiplicative":
                            dopant_term_multiplier = torch.tanh(h_out)
                        else:
                            element_offset = h_out
                    # When V53-γ1 is on without HN, still need total_conc.
                    if needs_conc_dep and dopant_total_conc is None and \
                       dopant_specs is not None:
                        _, _, dopant_total_conc = self._build_hypernet_inputs(
                            dopant_specs, fused.device, return_total_conc=True)
                    # Phase 47 V15: concat per-row weighted descriptor onto
                    # fused before feeding the VC head. No separate gating.
                    if isinstance(h, BrouwerHeadVC) and self.inject_descriptor \
                       and dopant_specs is not None:
                        descriptors, weights = self._build_hypernet_inputs(
                            dopant_specs, fused.device)
                        # weighted sum over K cations: [B, K, 5] × [B, K] → [B, 5]
                        weighted_desc = (weights.unsqueeze(-1) * descriptors).sum(dim=1)
                        fused_for_vc = torch.cat([fused, weighted_desc], dim=-1)
                    break
            from src.models.brouwer_head_kkt import BrouwerHeadKKT as _BHKKT
            preds = []
            for col in self.target_cols:
                head = self.heads[col]
                if isinstance(head, _BHKKT):
                    # Phase 53 V53-η: KKT-Hardnet head (Kröger-Vink)
                    method_idx_v = method_idx
                    if (getattr(head, "use_method_offset", False)
                            and method_idx_v is None and process is not None):
                        method_idx_v = self.process_stream._method_bits_to_idx(process)
                    preds.append(head(fused, process, dopant_specs=dopant_specs,
                                      method_idx=method_idx_v))
                elif isinstance(head, BrouwerHeadVC):
                    preds.append(head(fused_for_vc, process, dopant_specs=dopant_specs,
                                      method_idx=method_idx, class_idx=class_idx,
                                      element_offset=element_offset,
                                      dopant_term_multiplier=dopant_term_multiplier,
                                      dopant_total_conc=dopant_total_conc,
                                      dopant_a_offset=dopant_a_offset,
                                      dopant_b_offset=dopant_b_offset))
                elif isinstance(head, BrouwerHeadPDR):
                    preds.append(head(fused, process))
                else:
                    preds.append(head(fused))
        elif self.head_type == "brouwer_dft":
            # V58: DFTAnchoredBrouwerHead (M9) for VC reads dft_features + process;
            # PDR head reads (fused, process); others read fused.
            if process is None:
                raise ValueError("brouwer_dft head requires `process` in _heads_from_fused.")
            from src.models.brouwer_head_dft import DFTAnchoredBrouwerHead as _M9
            dft_features = self._lookup_dft_features(dopant_specs, process)
            dft_valid = self.dft_stream.valid_mask(dft_features)
            preds = []
            for col in self.target_cols:
                head = self.heads[col]
                if isinstance(head, _M9):
                    preds.append(head(fused, dft_features, process, dft_valid=dft_valid))
                elif isinstance(head, BrouwerHeadPDR):
                    preds.append(head(fused, process))
                else:
                    preds.append(head(fused))
        else:
            preds = [self.heads[col](fused) for col in self.target_cols]
        out = torch.cat(preds, dim=-1)                          # [B, num_targets]
        # Phase 59 — V59-CALM: add the zero-gated Δ-residual to the VC/PDR column.
        # Δ = 0 at init (zero w_out) ⇒ out == V55-Ext at step 0. Path-separated:
        # the modalities never touched `fused` (the magnitude floor u₀). C2.
        if (self.calm_enabled and self.calm_residual is not None
                and process is not None and dopant_specs is not None):
            delta = self.calm_delta(fused, dopant_specs, process)   # [B, 1]
            calm_cols = [i for i, c in enumerate(self.target_cols)
                         if c in ("vacancy_concentration", "photo_dark_ratio")]
            if calm_cols:
                add = torch.zeros_like(out)
                for i in calm_cols:
                    add[:, i:i + 1] = delta
                out = out + add
        if process is not None and dopant_specs is not None:
            out = self._maybe_apply_physics_envelope(out, dopant_specs, process)
        return out

    def _get_gate_features(self, dopant_specs: list[str]) -> torch.Tensor:
        """
        Chemistry-only features for the MoE gating network.

        - If a DopantStream is wired, its output (16-d explicit per-element
          embedding + descriptor weighted sum) is used.
        - Otherwise, fall back to CompositionStream output (64-d XenonPy MLP).

        These vectors are full-periodic-table covered, so unseen elements
        get sensible routing based on chemistry similarity rather than
        element-name lookup.
        """
        if self.dopant_stream is not None:
            return self.dopant_stream(dopant_specs)
        return self.composition_stream(dopant_specs)

    def get_gate_weights(self, target_col: str) -> torch.Tensor | None:
        """Return cached gate weights from the last MoE forward pass for `target_col`."""
        if self.head_type != "moe":
            return None
        if target_col not in self.heads:
            return None
        return self.heads[target_col].get_gate_weights()

    def mc_predict(
        self,
        graph,
        dopant_specs: list[str],
        process: torch.Tensor,
        n_passes: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference (Gal & Ghahramani 2016).

        Keeps dropout active in prediction heads while running N stochastic
        forward passes.  The mean approximates the ensemble prediction;
        the std approximates epistemic uncertainty.

        Args:
            graph, dopant_specs, process: Same as forward().
            n_passes: Number of stochastic forward passes (default 20).

        Returns:
            mean: [B, num_targets] — ensemble-averaged prediction.
            std:  [B, num_targets] — per-sample uncertainty estimate.
        """
        # Switch to eval mode (disables BN running stats update, etc.)
        # then re-enable dropout only in prediction heads.
        self.eval()
        for head in self.heads.values():
            for m in head.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        preds = []
        # Pre-compute gate features once (deterministic, no dropout) for MoE
        gate_feats = self._get_gate_features(dopant_specs) \
            if self.head_type == "moe" else None
        with torch.no_grad():
            for _ in range(n_passes):
                emb = self.get_embedding(graph, dopant_specs, process)
                preds.append(self._heads_from_fused(
                    emb, gate_features=gate_feats,
                    dopant_specs=dopant_specs, process=process,
                ))

        preds = torch.stack(preds, dim=0)   # [n_passes, B, T]
        mean = preds.mean(dim=0)            # [B, T]
        std  = preds.std(dim=0)             # [B, T]
        return mean, std

    def init_composition_decoder(self, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Initialise a composition reconstruction decoder (semi-supervised auxiliary loss).

        Must be called AFTER composition_stream.fit_scaler() so that raw_dim is known.
        The decoder reconstructs scaled XenonPy features from the fused embedding,
        providing gradient signal from all samples (including those with no regression
        labels). See ASGN (KDD 2020) and Hierarchical Mol Graph SSL (Comm. Chem. 2023).

        Args:
            hidden_dim: Hidden layer size for the decoder MLP.
            dropout   : Dropout rate in the decoder.
        """
        raw_dim = self.composition_stream.raw_dim
        fused_dim = sum(
            p.shape[0] for p in self.heads[self.target_cols[0]].net[0].parameters()
            if p.dim() == 2
        ) if self.target_cols else 160
        # Reliably get fused_dim from fusion output or fallback to 160
        try:
            # Access the first Linear in any head to get its in_features
            first_head = next(iter(self.heads.values()))
            fused_dim = first_head.net[0].in_features
        except Exception:
            fused_dim = 160

        self.composition_decoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, raw_dim),
        )
        # Move to same device as the rest of the model
        device = next(self.parameters()).device
        self.composition_decoder = self.composition_decoder.to(device)
        logger.info(
            f"Composition decoder initialised: {fused_dim}→{hidden_dim}→{raw_dim}"
        )

    def get_reconstruction(self, fused: torch.Tensor) -> torch.Tensor | None:
        """
        Reconstruct scaled XenonPy composition features from fused embedding.

        Returns None if the decoder has not been initialised via init_composition_decoder().

        Args:
            fused: [B, fused_dim] fused embedding tensor.

        Returns:
            [B, raw_dim] reconstructed composition features, or None.
        """
        decoder = getattr(self, "composition_decoder", None)
        if decoder is None:
            return None
        return decoder(fused)

    def freeze_encoder(self):
        """Freeze encoder parameters (Stage 2/3 mode)."""
        self.encoder.freeze()
        logger.info("Encoder frozen.")

    def unfreeze_encoder(self, lr_scale: float = 0.1):
        """
        Unfreeze encoder for fine-tuning with a small LR.
        Returns encoder parameters separately so the optimizer can apply
        a smaller learning rate.
        """
        self.encoder.unfreeze()
        logger.info("Encoder unfrozen.")

    def trainable_params_groups(
        self, base_lr: float, encoder_lr_scale: float = 0.1
    ) -> list[dict]:
        """
        Returns parameter groups for optimizer with separate LR for encoder.
        """
        encoder_params = list(self.encoder.parameters())
        dopant_params = (
            list(self.dopant_stream.parameters())
            if self.dopant_stream is not None else []
        )
        physics_params = []
        if self.physics_stream is not None:
            physics_params += list(self.physics_stream.parameters())
        if self.physics_envelope_pdr is not None:
            physics_params += list(self.physics_envelope_pdr.parameters())
        if self.physics_envelope_vc is not None:
            physics_params += list(self.physics_envelope_vc.parameters())
        aux_params = (
            list(self.aux_distill_head.parameters())
            if self.aux_distill_head is not None else []
        )
        other_params = (
            list(self.composition_stream.parameters())
            + list(self.process_stream.parameters())
            + dopant_params
            + physics_params
            + list(self.fusion.parameters())
            + [p for head in self.heads.values() for p in head.parameters()]
            + aux_params
        )
        return [
            {"params": encoder_params, "lr": base_lr * encoder_lr_scale},
            {"params": other_params, "lr": base_lr},
        ]

    @classmethod
    def from_pretrained(
        cls,
        encoder_ckpt: str,
        target_cols: list[str],
        fusion_mode: str = "concat",
        encoder_kwargs: dict | None = None,
        composition_kwargs: dict | None = None,
        process_kwargs: dict | None = None,
        dopant_kwargs: dict | None = None,
        encoder_backbone: str = "cgcnn",
        **kwargs,
    ) -> "Ga2O3Net":
        """
        Build Ga2O3Net by loading a pre-trained encoder from checkpoint.

        Args:
            encoder_ckpt: Path to pretrained_encoder.pt.
            target_cols: Target property names.
            fusion_mode: "concat" or "attention".
            encoder_backbone: "cgcnn" (default — Phase 43 V5 path) or
                "chgnet" (Phase 49 V17 — frozen 412k-param CHGNet pretrained
                on MPtrj 1.5M oxide structures; reads pre-extracted
                crystal_fea via batch.chgnet_emb).
            *_kwargs: Optional overrides for sub-module constructors.
            dopant_kwargs: When non-None, instantiate a DopantStream and wire
                it as the 4th fusion stream. Keys match DopantStream.__init__.
        """
        enc_kw = encoder_kwargs or {}
        comp_kw = composition_kwargs or {}
        proc_kw = process_kwargs or {}

        backbone = str(encoder_backbone).lower()
        if backbone == "chgnet":
            from src.models.chgnet_encoder import CHGNetCachedEncoder
            cache_path = enc_kw.get("chgnet_emb_cache_path")
            if cache_path is None:
                raise ValueError(
                    "encoder_backbone=chgnet requires "
                    "encoder_kwargs.chgnet_emb_cache_path "
                    "(produced by scripts/extract_chgnet_embeddings.py)"
                )
            encoder = CHGNetCachedEncoder(cache_path=cache_path)
            # CHGNet has no MP-pretrain_head equivalent — disable V5 distill aux
            if bool(kwargs.get("pretrain_distill_aux", False)):
                logger.warning(
                    "encoder_backbone=chgnet does not support pretrain_distill_aux; "
                    "disabling. Re-enable later by training a CHGNet→bandgap "
                    "auxiliary head if needed."
                )
                kwargs["pretrain_distill_aux"] = False
            logger.info("Using CHGNetCachedEncoder (frozen, "
                        f"cache={cache_path})")
        else:
            # Phase 43 V5: when distill aux is enabled, keep encoder.pretrain_head
            # alive so we can use its frozen output as a method-invariant aux target.
            keep_pretrain_head = bool(kwargs.get("pretrain_distill_aux", False))
            encoder = CGCNNEncoder(pretrain=keep_pretrain_head, **enc_kw)
            if os.path.exists(encoder_ckpt):
                state = torch.load(encoder_ckpt, map_location="cpu")
                if not keep_pretrain_head:
                    state = {k: v for k, v in state.items()
                             if not k.startswith("pretrain_head")}
                encoder.load_state_dict(state, strict=False)
                logger.info(f"Loaded encoder weights from {encoder_ckpt}"
                            + (" (incl. pretrain_head for distillation)"
                               if keep_pretrain_head else ""))
            else:
                logger.warning(f"Encoder checkpoint not found: {encoder_ckpt}. "
                               "Using random initialization.")

        composition = CompositionStream(**comp_kw)
        process = ProcessStream(**proc_kw)
        dopant = DopantStream(**dopant_kwargs) if dopant_kwargs else None

        model = cls(
            encoder=encoder,
            composition_stream=composition,
            process_stream=process,
            target_cols=target_cols,
            fusion_mode=fusion_mode,
            dopant_stream=dopant,
            **kwargs,
        )
        model.freeze_encoder()
        return model


class EnsembleGa2O3Net(nn.Module):
    """
    Inference-time ensemble wrapper over multiple Ga2O3Net members.

    All members share identical architecture and process / composition scalers
    (fitted on the same all-data view), but have independent weights from
    different random seeds — restoring the seed-diversity that 5-seed CV
    reports but that single-seed `train_final_model()` discards.

    forward()/mc_predict() average predictions across all members; mc_predict
    additionally runs n_passes MC-dropout draws *per member* and returns
    pooled mean / std, so both inter-seed (epistemic) and intra-seed
    (dropout) variance contribute to the uncertainty estimate.
    """

    def __init__(self, models: list[Ga2O3Net]):
        super().__init__()
        if not models:
            raise ValueError("EnsembleGa2O3Net needs at least one member.")
        self.members = nn.ModuleList(models)
        self.target_cols = models[0].target_cols

    def to(self, device):
        for m in self.members:
            m.to(device)
        return super().to(device)

    def eval(self):
        for m in self.members:
            m.eval()
        return super().eval()

    def forward(self, graph, dopant_specs, process):
        outs = [m(graph, dopant_specs, process) for m in self.members]
        return torch.stack(outs, dim=0).mean(dim=0)

    def mc_predict(
        self,
        graph,
        dopant_specs: list[str],
        process: torch.Tensor,
        n_passes: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_samples = []
        with torch.no_grad():
            for m in self.members:
                m.eval()
                for head in m.heads.values():
                    for sub in head.modules():
                        if isinstance(sub, nn.Dropout):
                            sub.train()
                gate_feats = m._get_gate_features(dopant_specs) \
                    if getattr(m, "head_type", "single") == "moe" else None
                for _ in range(n_passes):
                    emb = m.get_embedding(graph, dopant_specs, process)
                    all_samples.append(
                        m._heads_from_fused(
                            emb, gate_features=gate_feats,
                            dopant_specs=dopant_specs, process=process,
                        ))
        stacked = torch.stack(all_samples, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)

    def get_embedding(self, graph, dopant_specs, process):
        """Mean-pool fused embedding across members (used by 05_evaluate)."""
        embs = [m.get_embedding(graph, dopant_specs, process) for m in self.members]
        return torch.stack(embs, dim=0).mean(dim=0)
