"""
Stage 3: Few-shot fine-tuning trainer for Ga2O3Net.

Implements stratified k-fold cross-validation across all three model heads
(MLP and XGBoost), with correct data handling:

  * Scalers (process + composition) are fitted INSIDE each fold on training
    indices only — no data leakage from the validation set.
  * XGBoost is evaluated in the same fold loop, using fold-specific embeddings
    extracted after per-fold scaler fitting.
  * InfoNCE label classes are derived from the training fold's cation labels.

Data flow per fold:
  train_idx ─── fit scalers ──► model (frozen encoder)
                             ├─► MLP head training + early stopping
                             └─► XGBoost training (on extracted embeddings)
  val_idx ────────────────── ► MLP eval + XGBoost eval (leakage-free)
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.ga2o3_net import Ga2O3Net
from src.training.losses import CombinedLoss, UncertaintyWeightedLoss
from src.training.latent_synth_bank import LatentSynthBank
from src.data.experimental_dataset import (
    Ga2O3ExpDataset, PseudoLabeledSubset, collate_fn, TARGET_COLS,
)
from src.utils.metrics import regression_metrics

logger = logging.getLogger(__name__)


class FinetuneTrainer:
    """
    Few-shot fine-tuning with stratified k-fold cross-validation.

    Key design: all data-dependent fitting (scalers, LabelEncoder) happens
    inside each fold on training indices only.

    Args:
        model    : Ga2O3Net (initial weights; deep-copied per fold).
        dataset  : Ga2O3ExpDataset (no augmentation; augmented subset created per fold).
        config   : dict from fusion.yaml.
        device   : torch device.
        xgboost  : Whether to also run XGBoost in the same CV loop.
    """

    def __init__(
        self,
        model: Ga2O3Net,
        dataset: Ga2O3ExpDataset,
        config: dict,
        device: torch.device | str = "cpu",
        xgboost: bool = False,
    ):
        self.initial_model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        self.use_xgboost = xgboost
        self.use_gp = config.get("gp", {}).get("enabled", False)

        tc = config["training"]
        self.epochs = tc["epochs"]
        self.patience = tc["patience"]
        self.base_lr = tc["learning_rate"]
        self.encoder_lr = tc.get("encoder_lr", self.base_lr * 0.1)
        self.encoder_unfreeze_last_n = tc.get("encoder_unfreeze_last_n", 0)
        self.freeze_encoder = tc.get("freeze_encoder", True)
        self.grad_clip = tc.get("grad_clip", 1.0)
        self.weight_decay = tc.get("weight_decay", 0.0)
        # Phase 46 V11: element_hypernet gets its own weight_decay so the trainer
        # can apply L2 only to H without affecting α_method, encoder, or distill
        # aux head. Lit anchor: prevents H from absorbing α_method's contribution
        # under element/method correlation. Default falls back to self.weight_decay.
        self.hypernet_weight_decay = tc.get("hypernet_weight_decay", self.weight_decay)
        self.cv_folds = tc.get("cv_folds", 3)
        self.batch_size = tc.get("batch_size", 4)
        self.target_cols = config["targets"]

        # Optional improvements (backward-compatible: default to disabled)
        self.mc_dropout_n = tc.get("mc_dropout_passes", 0)
        self.mixup_alpha  = tc.get("mixup_alpha", 0.0)
        # Phase 27: C-Mixup label-aware pair sampling. When > 0, partners are
        # sampled by exp(-(y_i-y_j)^2 / (2·bw^2)) instead of uniform randperm.
        # Yao et al. NeurIPS 2022 (arXiv:2210.05775) shows vanilla mixup harms
        # regression; label-aware sampling fixes the linear-interpolation bias.
        self.mixup_label_kernel_bw = tc.get("mixup_label_kernel_bw", 0.0)
        self.mixup_primary_target_idx = tc.get("mixup_primary_target_idx", 0)
        self.pseudo_cfg   = config.get("pseudo_labeling", {})

        # Reconstruction auxiliary loss weight (0 = disabled)
        self.recon_weight = config.get("loss", {}).get("recon_weight", 0.0)

        lc = config["loss"]
        loss_type = lc.get("type", "combined")
        if loss_type == "uncertainty_weighted":
            self.loss_fn = UncertaintyWeightedLoss(
                num_tasks=len(self.target_cols),
                infonce_weight=lc["infonce_weight"],
                l2_lambda=lc["l2_lambda"],
                temperature=lc["infonce_temperature"],
                init_log_var=lc.get("init_log_var", 0.0),
                recon_weight=lc.get("recon_weight", 0.0),
            )
        else:
            self.loss_fn = CombinedLoss(
                mse_weight=lc.get("mse_weight", 0.85),
                infonce_weight=lc["infonce_weight"],
                l2_lambda=lc["l2_lambda"],
                temperature=lc["infonce_temperature"],
                recon_weight=lc.get("recon_weight", 0.0),
                moe_lb_weight=lc.get("moe_lb_weight", 0.0),
                regression_loss=lc.get("regression_loss", "mse"),
                huber_delta=lc.get("huber_delta", 1.0),
            )
        # Phase 6A: chemistry-family cos-similarity regularizer on DopantStream
        # embeddings. Pulls same-family dopant embeddings (e.g. Fe/Ta/Zr in
        # d-block, Si/Ge/Sn in IV) closer so rare-dopant classes borrow
        # representation from abundant same-family siblings.
        self.family_cos_weight = float(lc.get("family_cos_weight", 0.0))
        # Phase 6C: donor/acceptor monotonic-slope prior on dPDR/d(conc).
        # For each training row, look up the majority dopant's carrier_type:
        # +1 donor (Sn/Si/Ge/Ta/V/Zr/W/Mo/Nb/Sb/Ti/H) → enforce slope ≥ 0
        # -1 acceptor (Mg/Zn/Cu/Ni/Ca/Li/N/P)          → enforce slope ≤ 0
        #  0 isovalent / unknown                        → no constraint
        # Generalises to NEW dopants at deployment via pymatgen fallback in
        # src/data/element_descriptors.get_carrier_type.
        self.monotonic_weight = float(lc.get("monotonic_weight", 0.0))
        self.monotonic_delta  = float(lc.get("monotonic_delta", 0.15))
        # Monotonic loss is applied only to the PDR head; VC has no simple
        # donor/acceptor sign prior. `monotonic_target_idx` picks the column.
        self.monotonic_target_idx = int(lc.get("monotonic_target_idx", 0))
        # Phase 17: VC-specific monotonic prior — d(VC)/d(conc) ≥ 0 for any
        # cation substitutional dopant (Mg, Zn, Sn, Si, Ti, Ta, Nb, Cu, Fe, Cr,
        # La, Ce, ...). Universal physics rule from charge balance, applies to
        # unseen elements. Anion (N, F) and interstitial (H) excluded.
        self.monotonic_vc_weight = float(lc.get("monotonic_vc_weight", 0.0))
        self.monotonic_vc_delta  = float(lc.get("monotonic_vc_delta", 0.15))
        # Index of the VC column among the model's outputs (single-target VC = 0).
        self.monotonic_vc_target_idx = int(lc.get("monotonic_vc_target_idx", 0))
        # Phase 20: within-DOI pair-aware VC monotonic loss. Avoids cross-DOI
        # confound (e.g., high-conc Mg samples grown under O2-rich conditions
        # which suppress V_O empirically against physics). For pairs (i, j)
        # within the same batch with same DOI, same dopant element, and conc_i
        # < conc_j, require pred_VC[i] <= pred_VC[j].
        self.monotonic_vc_within_doi_weight = float(lc.get("monotonic_vc_within_doi_weight", 0.0))
        self.monotonic_vc_within_doi_min_dconc = float(lc.get("monotonic_vc_within_doi_min_dconc", 0.05))
        # V58: DFT-anchored BrouwerHead regularizers (doc §4.5).
        # dft_residual_reg: L2 on the bounded δ residual → stay near the DFT anchor.
        # tier1c_dft_consistency: pull unsupervised Tier-1C elements toward the pure
        # DFT anchor (the §1.1 OFFSET fix). Both no-op unless head is brouwer_dft.
        self.dft_residual_reg_weight = float(lc.get("dft_residual_reg_weight", 0.0))
        self.tier1c_dft_consistency_weight = float(lc.get("tier1c_dft_consistency_weight", 0.0))
        # Phase 33: when on, the within-DOI conc monotone uses element-class
        # sign (acceptor: −, donor: +, isovalent/anion: skipped) rather than
        # one-size-fits-all "more conc → more V_O" (which fights Mg/Zn data).
        self.monotonic_vc_within_doi_class_aware = bool(
            lc.get("monotonic_vc_within_doi_class_aware", False)
        )
        # Phase 32: atmosphere-ordinal soft-monotone VC loss (lit-backed sputter prior).
        # For pairs (i, j) with same dopant cation, |conc_i - conc_j| < dconc_max,
        # |T_i - T_j| < dT_max, and atm_ord_i < atm_ord_j (i more oxidising than j),
        # require pred_VC[i] + margin < pred_VC[j].
        self.atmo_ordinal_vc_weight       = float(lc.get("atmo_ordinal_vc_weight", 0.0))
        self.atmo_ordinal_vc_margin       = float(lc.get("atmo_ordinal_vc_margin", 0.10))
        self.atmo_ordinal_vc_max_dT       = float(lc.get("atmo_ordinal_vc_max_dT", 200.0))
        self.atmo_ordinal_vc_max_dconc    = float(lc.get("atmo_ordinal_vc_max_dconc", 0.5))  # log10 units
        self.atmo_ordinal_vc_within_doi_only = bool(lc.get("atmo_ordinal_vc_within_doi_only", False))
        # Phase 36: anneal-T sign-conditioned monotone for V_O.
        # In inert atmospheres (pO₂ < lo) Arrhenius diffusion → dV_O/dT > 0;
        # in oxidising atmospheres (pO₂ > hi) re-oxidation → dV_O/dT < 0.
        # Brouwer's closed form bakes in T+ universally; this loss SOFTLY
        # overrides for oxidising rows where the closed form is wrong.
        self.anneal_T_sign_weight = float(lc.get("anneal_T_sign_weight", 0.0))
        self.anneal_T_sign_delta  = float(lc.get("anneal_T_sign_delta", 50.0))     # °C perturbation
        self.anneal_T_sign_pO2_lo = float(lc.get("anneal_T_sign_pO2_lo", 0.20))    # below → inert (T+)
        self.anneal_T_sign_pO2_hi = float(lc.get("anneal_T_sign_pO2_hi", 0.50))    # above → oxidising (T−)

        # Phase 28-PDR: atmosphere-ordinal monotone for log[PDR] with INVERSE sign.
        self.atmo_ordinal_pdr_weight       = float(lc.get("atmo_ordinal_pdr_weight", 0.0))
        self.atmo_ordinal_pdr_margin       = float(lc.get("atmo_ordinal_pdr_margin", 0.20))
        self.atmo_ordinal_pdr_max_dT       = float(lc.get("atmo_ordinal_pdr_max_dT", 200.0))
        self.atmo_ordinal_pdr_max_dconc    = float(lc.get("atmo_ordinal_pdr_max_dconc", 0.5))

        # Phase 35: element-class contrastive prior on fused embeddings (lit P2).
        self.element_class_contrastive_weight = float(lc.get("element_class_contrastive_weight", 0.0))
        self.element_class_contrastive_margin = float(lc.get("element_class_contrastive_margin", 0.30))
        # Phase 23: cross-element BPR ranking loss. Within each batch, for pairs
        # (i, j) with different cation elements and both targets non-NaN,
        # use BPR log-sigmoid loss to encourage sign(pred_i - pred_j) ==
        # sign(true_i - true_j). Targets the cross-element ranking failure
        # (PDR ρ=0 on sputter) without forcing absolute directions.
        self.cross_elem_rank_weight = float(lc.get("cross_elem_rank_weight", 0.0))
        self.cross_elem_rank_target_idx = int(lc.get("cross_elem_rank_target_idx", 0))
        self.cross_elem_rank_min_dtrue = float(lc.get("cross_elem_rank_min_dtrue", 0.2))

        # Phase 42 V4: cross-method same-dopant fused-embedding PULL.
        # For each pair (i, j) in the batch with the SAME majority cation
        # but DIFFERENT deposition method, pull their fused embeddings
        # together via L2 distance. Hypothesis: same dopant_spec produces
        # method-invariant intrinsic chemistry (formation energy, bandgap,
        # ionic radius); pulling fused latent together gives the head a
        # cleaner element-specific dopant_term that is not contaminated by
        # method-specific baselines (those go to α_method[m] in BrouwerHeadVC).
        # Lit anchor: SIB-CL Loh 2022 Nat Commun (symmetry-pull → 9× data
        # efficiency); ACANet 2024 JCIM (per-pair-predicate triplet).
        self.cross_method_pull_weight = float(lc.get("cross_method_pull_weight", 0.0))

        # Phase 43 V5: distil pretrained encoder's pretrain_head outputs
        # (bandgap + formation_energy_per_atom) into a small linear head
        # over the fused embedding. Targets are method-INVARIANT chemistry,
        # so non-sputter rows shape only the chemistry-relevant subspace
        # of fused latent without polluting V_O/Brouwer scope.
        # Lit anchor: Crystal Twins (Magar 2022) + PretrainRoost (Huang
        # 2024 JCIM) — 10–27% MAE reduction in small-data regimes via
        # SSL/distill pretraining.
        self.encoder_distill_weight = float(lc.get("encoder_distill_weight", 0.0))

        # Phase 53 V53-ζ: PDR↔V_O latent InfoNCE contrastive regularizer.
        # Aligns fused embeddings of samples that share (DOI, dopant_label).
        # PDR has 52 labels, V_O has 14 — sharing latent axes raises the
        # effective N for V_O regularization to ~66 without enabling a PDR
        # regression head (V44-V8 multitask was NEG: PDR loss dilutes V_O).
        # Loss only fires when model.contrastive_proj is wired (via
        # multimodal.physics.contrastive_enabled = true).
        self.contrastive_pdr_vc_weight = float(
            lc.get("contrastive_pdr_vc_weight", 0.0)
        )
        self.contrastive_pdr_vc_temperature = float(
            lc.get("contrastive_pdr_vc_temperature", 0.5)
        )

        # Phase 54 V54-A1: SINCERE multi-positive InfoNCE + hard-neg + self-distill.
        # sincere_mode = "doi_label" preserves V53-ζ vanilla supervised InfoNCE on
        # (DOI, dopant_label) keys. "chem_sim" switches to SINCERE multi-positive
        # over chemistry buckets (Bi/Sb, Mg/Zn/Cu, Si/Sn/Ge, ...). "chem_sim+doi"
        # restricts positives to same-bucket AND same-DOI (safer if Mg specificity
        # collapses — Failure Path B in the plan).
        self.sincere_mode = str(lc.get("sincere_mode", "doi_label")).strip()
        if self.sincere_mode not in {"doi_label", "chem_sim", "chem_sim+doi"}:
            raise ValueError(
                f"Unknown sincere_mode={self.sincere_mode!r}; "
                "expected one of {'doi_label','chem_sim','chem_sim+doi'}"
            )
        # Hard-negative mining (Robinson et al. arXiv:2010.04592) — α=0 disables.
        # Only fires when sincere_mode != 'doi_label' so V53-ζ behavior is locked.
        self.hard_neg_alpha = float(lc.get("hard_neg_alpha", 0.0))
        # PDR-anchor SINCERE arm: anchor only PDR-labeled rows, but positives /
        # negatives drawn from full batch. Pushes 226 PDR labels to shape the
        # chemistry latent more aggressively than the symmetric V53-ζ arm does.
        self.pdr_anchor_weight = float(lc.get("pdr_anchor_weight", 0.0))
        # Self-distill from V53-ζ pseudo-targets on V_O-unlabeled rows.
        psd_cfg = lc.get("pdr_self_distill", {}) or {}
        if not isinstance(psd_cfg, dict):
            psd_cfg = {}
        self.self_distill_weight = float(psd_cfg.get("weight", 0.0))
        self.self_distill_path = str(
            psd_cfg.get("path", "data/processed/v53z_pseudo_targets_vo.npz")
        )
        self.self_distill_inv_var = bool(psd_cfg.get("inverse_variance", True))
        self.pseudo_vo_cache = None  # populated lazily by _load_pseudo_cache
        self.pseudo_std_cache = None

        # Phase 54 V54-A2: process-axis InfoNCE — third arm pulling fused
        # embeddings of (same dopant, same DOI, different process) together,
        # forcing chemistry to be process-invariant.
        self.process_axis_weight = float(lc.get("process_axis_weight", 0.0))
        self.process_axis_temperature = float(lc.get("process_axis_temperature", 0.5))

        # Phase 54 V54-C1: PySR symbolic dopant_term as soft constraint.
        # Compiles the closed-form expression discovered by
        # scripts/07_pysr_dopant_term.py into a torch.Tensor function;
        # adds MSE between ElementHyperNet output and PySR prediction per
        # element. Soft (Rule 1) — no hard substitution.
        pysr_cfg = lc.get("pysr", {}) or {}
        self.pysr_weight = float(pysr_cfg.get("weight", 0.0))
        self.pysr_path = str(pysr_cfg.get("path", "data/processed/pysr_dopant_term.json"))
        self.pysr_formula_compiled = None       # populated lazily by _load_pysr_formula

        # Phase 54 V54-B1: charge-state V_O / V_Ga frozen-expert distill weights.
        # Activated when the model's `charge_vo_expert` / `vga_expert` modules
        # are wired (via physics.charge_vo_expert_path / physics.vga_expert_path).
        self.charge_vo_distill_weight = float(lc.get("charge_vo_distill_weight", 0.0))
        self.vga_distill_weight       = float(lc.get("vga_distill_weight", 0.0))
        # V57-B1-Quartet+1: 4th DFT-scan expert (synthetic KROGER scan) distill.
        # Predicts log10[V_O^q] for q=0,+1,+2 from (elem, log_c, log_T, log_pO2);
        # Pulls aux_distill_dft_scan_head(fused) toward it. Roadmap caveat #7:
        # start at 0.05, halt grid if Mg@apsusc within-DOI flips.
        self.dft_scan_distill_weight = float(lc.get("dft_scan_distill_weight", 0.0))
        # V57-MACE-Latent: MACE-MP-0 → 64-d projection. Pulled into the
        # SINCERE InfoNCE positive-pair set as an auxiliary embedding signal.
        # Cleaner-than-MSE objective: cosine-similarity push between fused
        # latent (after a frozen projection) and the MACE projection.
        self.mace_latent_distill_weight = float(lc.get("mace_latent_distill_weight", 0.0))

        # Phase 53 V53-α: KROGER expert distill aux loss.
        # Pulls aux_distill_kroger_head(fused) toward kroger_expert(process,dopant)
        # — a closed-form β-Ga2O3 Brouwer prediction. Process-aware regularizer
        # that injects T/p_O2/concentration physics into the fused embedding.
        # Active when model.kroger_expert is wired and kroger_distill_weight > 0.
        self.kroger_distill_weight = float(lc.get("kroger_distill_weight", 0.0))

        # Phase 44 V8: SRH coupling — soft hinge penalising rows where BOTH
        # predicted V_O and predicted PDR are simultaneously high. Physics:
        # V_O are deep traps that increase SRH recombination, suppressing
        # PDR. The "high V_O AND high PDR" corner is unphysical. Penalty
        # only fires in that corner, so it cannot create slope sign flips.
        # Lit anchor: MEHnet (Nature Comp. Sci. 2025) — coupled-target
        # multi-task heads share representation across causally-linked
        # targets via soft physics coupling, not hard envelopes.
        self.srh_coupling_weight = float(lc.get("srh_coupling_weight", 0.0))
        self.srh_tau_vc          = float(lc.get("srh_tau_vc", 1.0))
        self.srh_tau_pdr         = float(lc.get("srh_tau_pdr", 1.0))

        # Phase 25-soft: physics envelope used as a **hinge loss prior** rather than
        # a multiplicative output transform. Penalises predictions exceeding the
        # physics-allowed ceiling env(λ, T) · log_PDR_max while leaving the head
        # free to learn true distribution. Envelope params (T_opt, σ_T, δ, σ_λ)
        # remain trainable so they converge to interpretable physics constants.
        self.physics_envelope_pdr_loss_weight = float(
            lc.get("physics_envelope_pdr_loss_weight", 0.0)
        )
        self.physics_envelope_pdr_target_idx = int(
            lc.get("physics_envelope_pdr_target_idx", 0)
        )
        self.physics_envelope_pdr_ceiling = float(
            lc.get("physics_envelope_pdr_ceiling_log10", 4.0)
        )
        self.physics_envelope_pdr_floor_log10 = float(
            lc.get("physics_envelope_pdr_floor_log10", 0.0)
        )

        # ── Phase 59 — V59-CALM Differentiable Counterfactual-Consistency (DCC).
        # Train-time, eval-symmetric version of scripts/eval_counterfactual_sweep.py:
        # for each graded element present in the batch, rebuild the dopant spec +
        # graph + process on a log-concentration grid (the SAME c_min/c_max/grid and
        # PROC_CONDITIONS the eval uses), forward through the model (incl. the CALM Δ),
        # and penalise WRONG-SIGN concentration steps with a one-sided squared hinge
        # (deadzone ε) per valence class (donor +, acceptor −, isovalent ≈flat). The
        # per-step finite-difference penalty simultaneously targets sign (LEARNED_LAW)
        # AND monotonicity (mono_frac), aggregated over process conditions (law_frac).
        # One-sided (lower bound on slope, never an upper clamp) ⇒ magnitude-safe:
        # it cannot compress dynamic range toward the mean (the Phase-24 envelope
        # failure). Sn guard (reduced q) protects the sputter_sn_vc r≥0.900 floor.
        self.dcc_weight       = float(lc.get("dcc_weight", 0.0))
        self.dcc_eps          = float(lc.get("dcc_eps", 0.02))      # deadzone (std units / grid step)
        self.dcc_n_grid       = int(lc.get("dcc_n_grid", 6))
        self.dcc_c_min        = float(lc.get("dcc_c_min", 1e-3))
        self.dcc_c_max        = float(lc.get("dcc_c_max", 8e-2))
        self.dcc_n_conditions = int(lc.get("dcc_n_conditions", 6))  # subset of the 6 eval conditions
        self.dcc_tau_flat     = float(lc.get("dcc_tau_flat", 0.05))
        self.dcc_isovalent_weight = float(lc.get("dcc_isovalent_weight", 0.5))
        self.dcc_warmup_frac  = float(lc.get("dcc_warmup_frac", 0.3))
        self.dcc_q_sn         = float(lc.get("dcc_q_sn", 0.3))       # Sn guard (reduced certainty)
        self.dcc_q_super      = float(lc.get("dcc_q_super", 0.6))    # Ta/W super-donors (sparser data)
        # Fire DCC every N optimizer steps over a FIXED graded element panel (decoupled
        # from batch composition, matching the eval probe set) → bounded cost. ~once/epoch.
        self.dcc_every_n_steps = int(lc.get("dcc_every_n_steps", 16))
        self.dcc_elements = lc.get("dcc_elements", None)   # None → default graded panel
        self.dcc_graph_cache  = {}   # spec_str → PyG graph (built once)
        self.dcc_proc_cache   = {}   # (meth,atm,T,c) → process tensor (built once)
        self._dcc_step = 0

        # Target standardization (per-fold z-score for better gradient conditioning)
        self.target_standardize = config.get("data", {}).get(
            "target_standardize", False
        )

        # Preserve initial state for per-fold reset
        self._initial_loss_fn = copy.deepcopy(self.loss_fn)

        # ── Phase 12A: optional latent synthesis bank ──────────────────────
        # config['latent_synth'] = {
        #   'bank_path': 'data/processed/phase12_synth.npz',
        #   'synth_weight': 0.3,    # sample_weight for synth rows
        #   'synth_loss_weight': 1.0,  # multiplier for synth-loss term added to total
        #   'synth_per_step': 8,    # number of synth rows mixed into each gradient step
        # }
        self.latent_synth_cfg = config.get("latent_synth")
        self.synth_bank = None
        if self.latent_synth_cfg and self.latent_synth_cfg.get("bank_path"):
            # Single-target session: pick the right column
            tgt = self.target_cols[0] if len(self.target_cols) == 1 else "photo_dark_ratio"
            self.synth_bank = LatentSynthBank(
                npz_path=self.latent_synth_cfg["bank_path"],
                target_col=tgt,
                synth_weight=self.latent_synth_cfg.get("synth_weight", 0.3),
            )
            self.synth_per_step = int(self.latent_synth_cfg.get("synth_per_step", 8))
            self.synth_loss_weight = float(
                self.latent_synth_cfg.get("synth_loss_weight", 1.0))
            self._cur_train_dois: set[str] | None = None    # set by _run_fold

    # ── Phase 6C: donor/acceptor monotonic-slope physics prior ───────────────
    def _monotonic_constraint_loss(self, model, batch) -> torch.Tensor:
        """Hinge loss enforcing sign(dPDR/dc) = carrier_type per row.

        Perturbs the process-tensor's log10_conc slot (index 11) by
        ``monotonic_delta`` (~+0.15 log10 ≈ ×1.4 concentration), forwards the
        model again, and penalises any direction mismatch with the carrier-
        type prior. Works for any dopant whose element symbol resolves via
        ``element_descriptors.get_carrier_type`` (includes pymatgen fallback
        for novel elements at deployment).

        Returns 0 tensor if ``monotonic_weight <= 0`` or no row in the batch
        has a non-zero carrier_type.
        """
        if self.monotonic_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.element_descriptors import get_carrier_type
        from src.data.dopant_spec import DopantSpec

        dopant_specs = batch["dopant_spec"]
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)

        carrier_vals: list[float] = []
        for spec_str in dopant_specs:
            parsed = DopantSpec.parse(spec_str)
            if parsed.is_undoped or not parsed.components:
                carrier_vals.append(0.0)
                continue
            majority = max(parsed.components, key=lambda c: c.conc).cation
            carrier_vals.append(get_carrier_type(majority))
        carrier = torch.tensor(carrier_vals, dtype=torch.float32, device=self.device)
        active = carrier.abs() > 0.5
        if not active.any():
            return torch.zeros((), device=self.device)

        emb_orig = model.get_embedding(graph, dopant_specs, process)
        pred_orig = model._heads_from_fused(emb_orig, dopant_specs=dopant_specs, process=process)

        process_pert = process.clone()
        process_pert[:, 11] = process_pert[:, 11] + self.monotonic_delta
        emb_pert = model.get_embedding(graph, dopant_specs, process_pert)
        pred_pert = model._heads_from_fused(emb_pert, dopant_specs=dopant_specs, process=process_pert)

        k = self.monotonic_target_idx
        dy = pred_pert[:, k] - pred_orig[:, k]                # [B]
        hinge = torch.clamp(-carrier * dy, min=0.0)           # [B]
        hinge = torch.where(active, hinge, torch.zeros_like(hinge))
        loss = hinge.sum() / active.sum().clamp(min=1).float()
        return self.monotonic_weight * loss

    def _monotonic_vc_concentration_loss(self, model, batch) -> torch.Tensor:
        """Phase 17: hinge loss enforcing d(VC)/d(log10_conc) ≥ 0 for cation
        substitutional dopants. Universal physics: charge-balance requires V_O
        to scale with cation dopant amount, regardless of donor/acceptor.

        Perturbs the process-tensor's log10_conc slot (index 11) by
        ``monotonic_vc_delta`` (~+0.15 log10 ≈ ×1.4 concentration), forwards
        the model again, and penalises any *negative* slope.

        Returns 0 if ``monotonic_vc_weight <= 0`` or no row in the batch is a
        cation substitutional dopant. Anion / interstitial dopants (N, F, H)
        and undoped rows are excluded.
        """
        if self.monotonic_vc_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.element_descriptors import is_cation_substitutional
        from src.data.dopant_spec import DopantSpec

        dopant_specs = batch["dopant_spec"]
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)

        is_cat: list[float] = []
        for spec_str in dopant_specs:
            parsed = DopantSpec.parse(spec_str)
            if parsed.is_undoped or not parsed.components:
                is_cat.append(0.0); continue
            majority = max(parsed.components, key=lambda c: c.conc).cation
            is_cat.append(1.0 if is_cation_substitutional(majority) else 0.0)
        mask = torch.tensor(is_cat, dtype=torch.float32, device=self.device)
        if mask.sum() < 1:
            return torch.zeros((), device=self.device)

        emb_orig = model.get_embedding(graph, dopant_specs, process)
        pred_orig = model._heads_from_fused(emb_orig, dopant_specs=dopant_specs, process=process)

        process_pert = process.clone()
        process_pert[:, 11] = process_pert[:, 11] + self.monotonic_vc_delta
        emb_pert = model.get_embedding(graph, dopant_specs, process_pert)
        pred_pert = model._heads_from_fused(emb_pert, dopant_specs=dopant_specs, process=process_pert)

        k = self.monotonic_vc_target_idx
        dy = pred_pert[:, k] - pred_orig[:, k]                 # [B], want ≥ 0
        hinge = torch.clamp(-dy, min=0.0) * mask               # zero for anion / undoped
        loss = hinge.sum() / mask.sum().clamp(min=1).float()
        return self.monotonic_vc_weight * loss

    def _monotonic_vc_within_doi_loss(self, model, batch) -> torch.Tensor:
        """Phase 20: within-DOI pair-aware VC monotonic loss.

        For each pair (i, j) in the batch with:
          - same DOI
          - same majority cation element
          - both classified as cation_substitutional
          - conc_i < conc_j (with min_dconc gap on log10 scale)

        require pred_VC[i] <= pred_VC[j]. Hinge: max(0, pred[i] - pred[j])^2.

        This avoids the cross-DOI confound where high-conc samples were grown
        under different O2 conditions, by only enforcing physics within a
        single paper's measurement series.
        """
        if self.monotonic_vc_within_doi_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.element_descriptors import is_cation_substitutional
        from src.data.dopant_spec import DopantSpec

        specs = batch["dopant_spec"]
        dois = batch.get("doi", [""] * len(specs))
        process = batch["process"].to(self.device)

        # Pre-extract per-sample (doi, element, conc, is_cation)
        n = len(specs)
        meta = []
        for i in range(n):
            doi_i = dois[i] if i < len(dois) else ""
            if not doi_i or doi_i == "—" or doi_i == "":
                meta.append(None); continue
            spec = DopantSpec.parse(specs[i])
            if spec.is_undoped or not spec.components:
                meta.append(None); continue
            elem = max(spec.components, key=lambda c: c.conc).cation
            if not is_cation_substitutional(elem):
                meta.append(None); continue
            meta.append((doi_i, elem, float(process[i, 11].detach())))

        # Phase 33: element-class direction map. Empirical sputter ground truth
        # (Mg ρ=−0.45, Sn ρ=+0.93) shows acceptors and donors take OPPOSITE
        # signs vs concentration — defect-defect complex formation (Mg-V_O
        # association reduces measurable V_O; Sn-V_O charge-compensation
        # increases V_O). When ``class_aware`` is on, the loss flips sign for
        # acceptor elements rather than enforcing one-size-fits-all.
        if self.monotonic_vc_within_doi_class_aware:
            from src.models.physics_features import _ELEM_PROPS, GA_VALENCE
            def elem_direction(elem: str) -> int:
                p = _ELEM_PROPS.get(elem)
                if p is None or p.get("is_anion", False):
                    return 0  # skip anions / unknowns
                v = p["valence"]
                if v < GA_VALENCE: return -1   # acceptor: more conc → less V_O
                if v > GA_VALENCE: return +1   # donor:    more conc → more V_O
                return 0                        # isovalent: no constraint
        else:
            def elem_direction(elem: str) -> int:
                return +1   # legacy Phase 20 behaviour

        # Find within-DOI same-element pairs
        pairs = []
        for i in range(n):
            if meta[i] is None: continue
            doi_i, elem_i, conc_i = meta[i]
            for j in range(i + 1, n):
                if meta[j] is None: continue
                doi_j, elem_j, conc_j = meta[j]
                if doi_j != doi_i or elem_j != elem_i: continue
                if abs(conc_i - conc_j) < self.monotonic_vc_within_doi_min_dconc: continue
                direction = elem_direction(elem_i)
                if direction == 0:
                    continue
                if conc_i < conc_j:
                    pairs.append((i, j, direction))   # (low_conc, high_conc, dir)
                else:
                    pairs.append((j, i, direction))

        if not pairs:
            return torch.zeros((), device=self.device)

        # Forward pass on whole batch (one model call), gather pair predictions
        graph = batch["graph"].to(self.device)
        emb = model.get_embedding(graph, specs, process)
        pred = model._heads_from_fused(emb, dopant_specs=specs, process=process)
        k = self.monotonic_vc_target_idx

        hinges = []
        for low_conc, high_conc, direction in pairs:
            if direction > 0:
                # donor: pred[high_conc] >= pred[low_conc] (more conc → more V_O)
                h = torch.clamp(pred[low_conc, k] - pred[high_conc, k], min=0.0)
            else:
                # acceptor: pred[high_conc] <= pred[low_conc] (more conc → less V_O)
                h = torch.clamp(pred[high_conc, k] - pred[low_conc, k], min=0.0)
            hinges.append(h * h)   # squared hinge for smoother gradient
        return self.monotonic_vc_within_doi_weight * torch.stack(hinges).mean()

    def _dcc_loss(self, model, batch) -> torch.Tensor:
        """Phase 59 — V59-CALM Differentiable Counterfactual-Consistency (DCC).

        For each graded element present in this batch, sweep ONLY concentration on a
        log-grid (rebuilding spec+graph+process exactly like
        scripts/eval_counterfactual_sweep.py) at several PROC_CONDITIONS, and apply a
        one-sided squared-hinge on each consecutive finite-difference step:

            donor/acceptor :  q_e · Σ_k max(0, −s_e·(ŷ_{k+1}−ŷ_k) + ε)²
            isovalent      :  w_iso · Σ_k max(0, |ŷ_{k+1}−ŷ_k| − τ_flat)²

        s_e = +1 donor / −1 acceptor (signs invert for PDR). One-sided ⇒ never an
        upper clamp ⇒ magnitude-safe (Rule 2; the closed-form Brouwer floor + Δ keep
        the magnitude). Gradient flows to the head's dopant_term/ΔE_f and the CALM Δ
        only — modalities stay detached (Rule 1).
        """
        if self.dcc_weight <= 0.0:
            return torch.zeros((), device=self.device)
        # Fire only every N optimizer steps (bounded cost; ~once per epoch).
        self._dcc_step += 1
        if self.dcc_every_n_steps > 1 and (self._dcc_step % self.dcc_every_n_steps) != 0:
            return torch.zeros((), device=self.device)
        # warm-up: let magnitude settle before engaging the law term (doc §8).
        ep = getattr(self, "_cur_epoch", None)
        tot = getattr(self, "_cur_total_epochs", None)
        warm = 1.0
        if ep is not None and tot and self.dcc_warmup_frac > 0:
            warm = min(1.0, ep / max(1.0, self.dcc_warmup_frac * tot))
        if warm <= 0.0:
            return torch.zeros((), device=self.device)

        from src.data.dopant_spec import DopantSpec
        from src.data.graph_builder import get_graph_for_spec
        from src.data.experimental_dataset import build_process_tensor
        from torch_geometric.data import Batch
        from scripts.eval_physics_diag_table import (
            ACCEPTOR_ELEMENTS, DONOR_ELEMENTS, SUPER_DONOR_ELEMENTS, ISOVALENT_ELEMENTS,
        )
        import numpy as np

        # Which single target is this run (DCC is single-target like the rest of V59)?
        tcols = list(model.target_cols)
        tgt = None
        for cand in ("vacancy_concentration", "photo_dark_ratio"):
            if cand in tcols:
                tgt = cand; tgt_idx = tcols.index(cand); break
        if tgt is None:
            return torch.zeros((), device=self.device)
        flip = (tgt == "photo_dark_ratio")

        def _sign_q(elem):
            if elem in ACCEPTOR_ELEMENTS:
                return (+1 if flip else -1), 1.0
            if elem in DONOR_ELEMENTS:
                q = self.dcc_q_sn if elem == "Sn" else 1.0   # Sn guard
                return (-1 if flip else +1), q
            if elem in SUPER_DONOR_ELEMENTS:
                return (-1 if flip else +1), self.dcc_q_super
            if elem in ISOVALENT_ELEMENTS:
                return 0, 1.0
            return None, 0.0   # unknown → q=0 (never assert a sign we're unsure of)

        # FIXED graded element panel (decoupled from the batch, matching the eval probe
        # set): all donors/acceptors/super-donors + a few isovalent controls. DCC is a
        # global concentration-law regularizer, not a per-row term.
        if self.dcc_elements:
            elems = list(self.dcc_elements)
        else:
            elems = sorted(set(ACCEPTOR_ELEMENTS) | set(DONOR_ELEMENTS)
                           | set(SUPER_DONOR_ELEMENTS) | {"Al", "Fe"})

        PROC_CONDITIONS = [
            ("RF magnetron sputtering", "Ar", 500.0),
            ("RF magnetron sputtering", "Ar", 900.0),
            ("RF magnetron sputtering", "O2", 700.0),
            ("RF magnetron sputtering", "Ar:O2=1:1", 700.0),
            ("MOCVD", "O2", 700.0),
            ("PLD", "Ar", 600.0),
        ][: max(1, self.dcc_n_conditions)]
        grid = np.logspace(np.log10(self.dcc_c_min), np.log10(self.dcc_c_max), self.dcc_n_grid)

        def _graph(spec_str):
            g = self.dcc_graph_cache.get(spec_str)
            if g is None:
                g = get_graph_for_spec(DopantSpec.parse(spec_str))
                self.dcc_graph_cache[spec_str] = g
            return g

        def _proc(meth, atm, T, c):
            key = (meth, atm, T, round(float(c), 6))
            p = self.dcc_proc_cache.get(key)
            if p is None:
                p = build_process_tensor(temperature_C=T, time_min=60.0, atmosphere=atm,
                                         method=meth, concentration_total_frac=float(c))
                self.dcc_proc_cache[key] = p
            return p

        terms = []
        eps, tau = self.dcc_eps, self.dcc_tau_flat
        for elem in elems:
            s_e, q_e = _sign_q(elem)
            if q_e <= 0.0:
                continue
            # Build the per-concentration graphs once (process-independent).
            try:
                specs_k = [f"{elem}:{c:.6f}" for c in grid]
                graphs_k = [_graph(sp) for sp in specs_k]
            except Exception:
                continue
            for (meth, atm, T) in PROC_CONDITIONS:
                procs_k = torch.stack([_proc(meth, atm, T, c) for c in grid]).to(self.device)
                gbatch = Batch.from_data_list(graphs_k).to(self.device)
                emb = model.get_embedding(gbatch, specs_k, procs_k)
                pred = model._heads_from_fused(emb, dopant_specs=specs_k, process=procs_k)[:, tgt_idx]
                d = pred[1:] - pred[:-1]            # [K-1] consecutive log-conc steps
                if s_e == 0:
                    # isovalent: penalise any strong slope (one-sided on |d|).
                    h = torch.clamp(d.abs() - tau, min=0.0)
                    terms.append(self.dcc_isovalent_weight * (h * h).mean())
                else:
                    # donor/acceptor: one-sided wrong-direction hinge with deadzone ε.
                    h = torch.clamp(-float(s_e) * d + eps, min=0.0)
                    terms.append(q_e * (h * h).mean())

        if not terms:
            return torch.zeros((), device=self.device)
        return self.dcc_weight * warm * torch.stack(terms).mean()

    def _element_class_contrastive_loss(self, model, batch) -> torch.Tensor:
        """Phase 35: element-class contrastive prior on FUSED EMBEDDINGS.

        Lit-recommended P2 prior (agent scan 2026-04-30): pull embeddings of
        same-class dopants together, push different-class apart. Uses the same
        valence-based class map as Phase 33 (acceptor v<3, donor v>3,
        isovalent v=3 — which we treat as a third cluster instead of skipping
        because it has many elements: Al, Cr, Fe, Mn, In, Y, La, B…).

        For each anchor-positive-negative triplet within batch:
            L = max(0, ||emb_a − emb_p||² − ||emb_a − emb_n||² + margin)

        Regularises the encoder/fusion to learn class structure → should help
        OOD generalisation to rare elements (Cu/Cr/Mn/Bi etc. with ≤2 samples).
        """
        if self.element_class_contrastive_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.dopant_spec import DopantSpec
        from src.models.physics_features import _ELEM_PROPS, GA_VALENCE

        specs = batch["dopant_spec"]
        n = len(specs)
        if n < 3:
            return torch.zeros((), device=self.device)

        # Element class lookup: 0=acceptor, 1=isovalent, 2=donor; -1 = skip
        def _class_of(elem: str) -> int:
            p = _ELEM_PROPS.get(elem)
            if p is None or p.get("is_anion", False): return -1
            v = p["valence"]
            if v < GA_VALENCE: return 0   # acceptor
            if v > GA_VALENCE: return 2   # donor
            return 1                       # isovalent

        cls = []
        for i in range(n):
            spec = DopantSpec.parse(specs[i])
            if spec.is_undoped or not spec.components:
                cls.append(-1); continue
            elem = max(spec.components, key=lambda c: c.conc).cation
            cls.append(_class_of(elem))

        # Build all valid triplets (anchor, positive, negative) in this batch
        triplets = []
        for a in range(n):
            if cls[a] < 0: continue
            same = [j for j in range(n) if j != a and cls[j] == cls[a]]
            diff = [j for j in range(n) if j != a and cls[j] >= 0 and cls[j] != cls[a]]
            if not same or not diff: continue
            # Take first valid (p, n) — keeps O(n) instead of O(n³)
            triplets.append((a, same[0], diff[0]))

        if len(triplets) < 1:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        emb = model.get_embedding(graph, specs, process)   # [B, fused_dim]
        # Normalise per row so distance is in [0, 4] regardless of model scale
        emb_n = torch.nn.functional.normalize(emb, dim=-1)

        m = self.element_class_contrastive_margin
        losses = []
        for a, p, q in triplets:   # q = negative
            d_ap = ((emb_n[a] - emb_n[p]) ** 2).sum()
            d_an = ((emb_n[a] - emb_n[q]) ** 2).sum()
            losses.append(torch.clamp(d_ap - d_an + m, min=0.0))
        return self.element_class_contrastive_weight * torch.stack(losses).mean()

    def _atmosphere_ordinal_pdr_loss(self, model, batch) -> torch.Tensor:
        """Phase 28-PDR: atmosphere-ordinal soft-monotone log[PDR] loss.

        Sign INVERSE of VC version: more oxidising (lower atm_ord) → fewer V_O →
        less dark current → HIGHER PDR. So for pairs (i, j) with the same dopant
        and similar T/conc, atm_ord_i < atm_ord_j (i more oxidising) implies
        ``pred_PDR[i] − pred_PDR[j] ≥ margin``.
        """
        if self.atmo_ordinal_pdr_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.dopant_spec import DopantSpec
        from src.data.experimental_dataset import parse_atmosphere_ordinal

        specs = batch["dopant_spec"]
        atms  = batch.get("atmosphere", [""] * len(specs))
        process = batch["process"].to(self.device)

        n = len(specs)
        meta = []
        for i in range(n):
            atm_ord = parse_atmosphere_ordinal(str(atms[i] if i < len(atms) else ""))
            T_i = float(process[i, 0].detach())
            log_c_i = float(process[i, 11].detach()) if process.shape[1] >= 12 else 0.0
            spec = DopantSpec.parse(specs[i])
            if spec.is_undoped or not spec.components:
                elem = "_undoped"
            else:
                elem = max(spec.components, key=lambda c: c.conc).cation
            meta.append((elem, atm_ord, T_i, log_c_i))

        pairs = []
        for i in range(n):
            el_i, ord_i, T_i, lc_i = meta[i]
            for j in range(i + 1, n):
                el_j, ord_j, T_j, lc_j = meta[j]
                if el_j != el_i: continue
                if ord_i == ord_j: continue
                if abs(T_i - T_j) > self.atmo_ordinal_pdr_max_dT: continue
                if abs(lc_i - lc_j) > self.atmo_ordinal_pdr_max_dconc: continue
                if ord_i < ord_j:   # i more oxidising → predict HIGHER PDR
                    pairs.append((i, j))   # (high_PDR, low_PDR)
                else:
                    pairs.append((j, i))

        if not pairs:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        emb = model.get_embedding(graph, specs, process)
        pred = model._heads_from_fused(emb, dopant_specs=specs, process=process)
        # PDR target index — for shared-head models, photo_dark_ratio is the first
        # column when both targets are present.
        try:
            k = self.target_cols.index("photo_dark_ratio")
        except ValueError:
            k = 0
        m = self.atmo_ordinal_pdr_margin

        hinges = []
        for high_pdr, low_pdr in pairs:
            # want pred[high_pdr] − pred[low_pdr] ≥ m
            h = torch.clamp(m - (pred[high_pdr, k] - pred[low_pdr, k]), min=0.0)
            hinges.append(h * h)
        return self.atmo_ordinal_pdr_weight * torch.stack(hinges).mean()

    def _anneal_temperature_sign_loss(self, model, batch) -> torch.Tensor:
        """Phase 36: anneal-T sign-conditioned monotone V_O loss.

        Brouwer head closed form forces ``log[V_O] ∝ −E_f/(kT·ln10)`` —
        monotone-increasing in T regardless of atmosphere. The OA sputter
        β-Ga₂O₃ literature (Onuma 2015, Higashiwaki 2017) shows the sign
        depends on pO₂:

          - inert (Ar/vacuum/N₂, pO₂ < ``pO2_lo``):  dV_O/dT > 0  (Arrhenius)
          - oxidising (O₂/O₂_plasma, pO₂ > ``pO2_hi``): dV_O/dT < 0  (re-oxidation)

        For each batch row we perturb ``process[:, 0]`` (temperature_C) by
        ``+anneal_T_sign_delta`` °C, forward the model twice, and penalise
        when sign(dV_O/dT) disagrees with the atmosphere-driven target sign.
        Mixed atmospheres (lo ≤ pO₂ ≤ hi) are skipped (no constraint).

        Soft override of Brouwer's hard T+ — Brouwer wins on inert (sign
        agrees, hinge always 0) but the loss pulls oxidising rows toward
        T− where the closed form is wrong.
        """
        if self.anneal_T_sign_weight <= 0.0:
            return torch.zeros((), device=self.device)

        process = batch["process"].to(self.device)
        o2 = process[:, 2]   # o2_fraction in [0, 1]

        # Per-row target sign: +1 inert, −1 oxidising, 0 mixed (skip)
        sign = torch.zeros_like(o2)
        sign = torch.where(o2 < self.anneal_T_sign_pO2_lo, torch.ones_like(o2), sign)
        sign = torch.where(o2 > self.anneal_T_sign_pO2_hi, -torch.ones_like(o2), sign)
        active = sign != 0
        if active.sum() < 1:
            return torch.zeros((), device=self.device)

        dopant_specs = batch["dopant_spec"]
        graph = batch["graph"].to(self.device)

        emb_orig = model.get_embedding(graph, dopant_specs, process)
        pred_orig = model._heads_from_fused(emb_orig, dopant_specs=dopant_specs, process=process)

        process_pert = process.clone()
        process_pert[:, 0] = process_pert[:, 0] + self.anneal_T_sign_delta
        emb_pert = model.get_embedding(graph, dopant_specs, process_pert)
        pred_pert = model._heads_from_fused(emb_pert, dopant_specs=dopant_specs, process=process_pert)

        k = self.monotonic_vc_target_idx
        dy = pred_pert[:, k] - pred_orig[:, k]                  # [B]
        # Penalise when sign·dy ≤ 0 (wrong direction). Linear hinge.
        hinge = torch.clamp(-sign * dy, min=0.0)                # [B]
        hinge = torch.where(active, hinge, torch.zeros_like(hinge))
        loss = hinge.sum() / active.sum().clamp(min=1).float()
        return self.anneal_T_sign_weight * loss

    def _atmosphere_ordinal_vc_loss(self, model, batch) -> torch.Tensor:
        """Phase 32: atmosphere-ordinal soft-monotone V_O loss.

        OA β-Ga₂O₃ sputter literature consistently orders V_O density by
        atmosphere oxidising power:
          O₂_plasma < O₂ < Ar:O₂ mix < Ar+H₂ < Ar+N₂ < Ar < vacuum
        (mapped to ordinals 0..6 in ``parse_atmosphere_ordinal``).

        For each pair (i, j) in the batch with:
          - same majority cation element
          - |T_anneal_i − T_anneal_j| ≤ ``atmo_ordinal_vc_max_dT``
          - |log10(conc_i) − log10(conc_j)| ≤ ``atmo_ordinal_vc_max_dconc``
          - atm_ord_i < atm_ord_j (i more oxidising → expect less V_O)

        require pred_VC[j] − pred_VC[i] ≥ margin. Squared hinge.

        Complements Phase 20 within-DOI conc loss (which is conc-axis only)
        and Brouwer head's pO₂ term (which fails for plasma/H₂ rows where
        atomic-O activity is ill-defined). Soft prior — does not force
        magnitude, only ordering, so it cannot reproduce the Phase 24v2
        catastrophe.
        """
        if self.atmo_ordinal_vc_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.dopant_spec import DopantSpec
        from src.data.experimental_dataset import parse_atmosphere_ordinal

        specs = batch["dopant_spec"]
        dois  = batch.get("doi", [""] * len(specs))
        atms  = batch.get("atmosphere", [""] * len(specs))
        process = batch["process"].to(self.device)

        n = len(specs)
        meta = []
        for i in range(n):
            if i >= len(atms): meta.append(None); continue
            atm_ord = parse_atmosphere_ordinal(str(atms[i]))
            T_i = float(process[i, 0].detach())          # temperature_C
            log_c_i = float(process[i, 11].detach()) if process.shape[1] >= 12 else 0.0
            spec = DopantSpec.parse(specs[i])
            if spec.is_undoped or not spec.components:
                elem = "_undoped"
            else:
                elem = max(spec.components, key=lambda c: c.conc).cation
            doi_i = dois[i] if i < len(dois) else ""
            meta.append((doi_i, elem, atm_ord, T_i, log_c_i))

        # Find valid pairs
        pairs = []
        for i in range(n):
            if meta[i] is None: continue
            doi_i, el_i, ord_i, T_i, lc_i = meta[i]
            for j in range(i + 1, n):
                if meta[j] is None: continue
                doi_j, el_j, ord_j, T_j, lc_j = meta[j]
                if el_j != el_i: continue
                if self.atmo_ordinal_vc_within_doi_only and doi_i != doi_j: continue
                if ord_i == ord_j: continue
                if abs(T_i - T_j) > self.atmo_ordinal_vc_max_dT: continue
                if abs(lc_i - lc_j) > self.atmo_ordinal_vc_max_dconc: continue
                if ord_i < ord_j:   # i more oxidising → predict lower V_O
                    pairs.append((i, j))   # (low_VO, high_VO)
                else:
                    pairs.append((j, i))

        if not pairs:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        emb = model.get_embedding(graph, specs, process)
        pred = model._heads_from_fused(emb, dopant_specs=specs, process=process)
        k = self.monotonic_vc_target_idx
        m = self.atmo_ordinal_vc_margin

        hinges = []
        for low, high in pairs:
            # want pred[high] - pred[low] >= m  →  penalize m - (pred[high] - pred[low])
            h = torch.clamp(m - (pred[high, k] - pred[low, k]), min=0.0)
            hinges.append(h * h)
        return self.atmo_ordinal_vc_weight * torch.stack(hinges).mean()

    def _cross_method_dopant_pull_loss(self, model, batch) -> torch.Tensor:
        """Phase 42 V4: pull fused embeddings together for cross-method
        same-dopant pairs.

        For pair (i, j) in the batch with:
          - same majority cation (e.g., both Mg, both Sn)
          - different deposition method (e.g., one sputter, one CVD/ALD)
          - both labels valid (avoid pulling to noise)

        compute  L = ||fused[i] - fused[j]||² / dim
        and average over pairs.

        Physics rationale: a Mg-doped β-Ga₂O₃ has the same intrinsic
        formation energy, bandgap, ionic-radius mismatch regardless of
        whether it was sputtered or CVD-grown. The METHOD-specific
        differences (V_O baseline, cross-DOI experimental noise) should
        live in α_method[m] inside BrouwerHeadVC, NOT in the fused
        latent. Pulling cross-method same-dopant fused embeddings together
        forces the latent to encode method-invariant chemistry only.

        Lit anchor: SIB-CL (Loh 2022 Nat Commun) — pull on symmetry-sharing
        pairs gives 9× target-data efficiency; ACANet (2024 JCIM) — explicit
        per-pair predicate triplet for activity cliffs.
        """
        if self.cross_method_pull_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.dopant_spec import DopantSpec

        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)

        # Method index from process tensor — same convention as ProcessStream
        # (continuous_dim:continuous_dim+5 are sputter/PLD/CVD-ALD/wet/evap;
        # all-zero → "other" idx 5). We rely on the model's process_stream
        # to compute it consistently with what the encoder sees.
        method_idx = model.process_stream._method_bits_to_idx(process)  # [B] long

        n = len(specs)
        if n < 2:
            return torch.zeros((), device=self.device)

        # Compute majority cation per row
        cations = []
        for i in range(n):
            spec = DopantSpec.parse(specs[i])
            if spec.is_undoped or not spec.components:
                cations.append("_undoped")
            else:
                cations.append(max(spec.components, key=lambda c: c.conc).cation)

        # Build cross-method same-cation pairs
        pairs = []
        for i in range(n):
            if cations[i] == "_undoped":
                # Undoped rows share no element; pulling them together would
                # blur method differences without physical meaning. Skip.
                continue
            for j in range(i + 1, n):
                if cations[j] != cations[i]:
                    continue
                if int(method_idx[i].item()) == int(method_idx[j].item()):
                    continue
                pairs.append((i, j))

        if not pairs:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        emb = model.get_embedding(graph, specs, process)  # [B, fused_dim]
        # Normalize by fused_dim so the L2 distance scale is comparable across
        # encoder/fusion configurations.
        fused_dim = emb.shape[-1]
        dists = []
        for (i, j) in pairs:
            d = (emb[i] - emb[j]).pow(2).sum() / float(fused_dim)
            dists.append(d)
        return self.cross_method_pull_weight * torch.stack(dists).mean()

    def _encoder_distill_loss(self, model, batch) -> torch.Tensor:
        """Phase 43 V5: distil pretrained encoder.pretrain_head outputs
        (bandgap + formation_energy_per_atom) into aux_distill_head over
        fused embedding.

        Targets are intrinsic CHEMISTRY (method-invariant by construction)
        so applying this loss on ALL rows — sputter and non-sputter alike
        — pulls the fused latent into the chemistry-relevant subspace
        without leaking method-specific signal into V_O scope.
        """
        if self.encoder_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "aux_distill_head", None) is None:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)

        # Method-invariant target: frozen encoder.pretrain_head over struct_emb.
        # Encoder is frozen (requires_grad=False); detach for safety.
        with torch.no_grad():
            struct_emb = model.encoder.get_embedding(graph)         # [B, 64]
            target = model.encoder.pretrain_head(struct_emb).detach()  # [B, 2]

        fused = model.get_embedding(graph, specs, process)          # [B, fused_dim]
        pred = model.aux_distill_head(fused)                         # [B, 2]

        loss = ((pred - target) ** 2).mean()
        return self.encoder_distill_weight * loss

    def _kroger_distill_loss(self, model, batch) -> torch.Tensor:
        """Phase 53 V53-α: distill encoder_kroger's log_VO prediction into fused.

        kroger_expert is frozen; aux_distill_kroger_head learns to predict its
        output from fused. This injects (T, p_O2, dopant, [c]) → log_VO physics
        into the fused latent on ALL rows (sputter + non-sputter).

        Loss is zero when model.kroger_expert is None or weight is 0.
        """
        if self.kroger_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "kroger_expert", None) is None:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)

        # Frozen expert prediction (target). Run on the same device.
        with torch.no_grad():
            target = model.kroger_expert(process, specs).detach()  # [B]
        # Standardize target to z-score (similar to what V_O regression does)
        # The kroger expert outputs raw log10[V_O] in [9, 22.5] range.
        # We z-score by training-set stats from the kroger checkpoint to match
        # the magnitudes of fused-derived predictions.
        target_norm = (target - model.kroger_expert.target_mean) / max(
            model.kroger_expert.target_std, 1e-3
        )

        fused = model.get_embedding(graph, specs, process)         # [B, fused_dim]
        pred = model.aux_distill_kroger_head(fused).squeeze(-1)    # [B]

        loss = ((pred - target_norm) ** 2).mean()
        return self.kroger_distill_weight * loss

    # ------------------------------------------------------------------
    # Phase 54 V54-A1 helpers
    # ------------------------------------------------------------------
    def _build_group_ids(self, batch, mode: str) -> torch.Tensor:
        """Build [B] group-id tensor for SINCERE / InfoNCE.

        mode = "doi_label":      keymap of (doi, dopant_label) — V53-ζ default
        mode = "chem_sim":       chemistry-bucket id only (cross-DOI positives)
        mode = "chem_sim+doi":   composite (chem_bucket, doi)
        """
        specs = batch["dopant_spec"]
        labels = batch.get("dopant_label", [""] * len(specs))
        if mode == "doi_label":
            dois = batch.get("doi", [""] * len(specs))
            keymap: dict[tuple, int] = {}
            ids_list: list[int] = []
            for d, l in zip(dois, labels):
                key = (str(d), str(l))
                if key not in keymap:
                    keymap[key] = len(keymap)
                ids_list.append(keymap[key])
            return torch.tensor(ids_list, dtype=torch.long, device=self.device)

        from src.data.element_descriptors import (
            chem_sim_group_ids_for_batch,
            CHEM_SIM_UNDOPED_SENTINEL,
        )
        chem_ids = chem_sim_group_ids_for_batch([str(l) for l in labels])
        if mode == "chem_sim":
            return torch.tensor(chem_ids, dtype=torch.long, device=self.device)
        if mode == "chem_sim+doi":
            dois = batch.get("doi", [""] * len(specs))
            keymap2: dict[tuple, int] = {}
            ids_list2: list[int] = []
            for c, d in zip(chem_ids, dois):
                if c == CHEM_SIM_UNDOPED_SENTINEL:
                    ids_list2.append(CHEM_SIM_UNDOPED_SENTINEL)
                    continue
                key = (int(c), str(d))
                if key not in keymap2:
                    keymap2[key] = len(keymap2)
                ids_list2.append(keymap2[key])
            return torch.tensor(ids_list2, dtype=torch.long, device=self.device)
        raise ValueError(f"Unknown sincere_mode={mode!r}")

    def _build_chem_emb_for_batch(self, dopant_labels: list[str]) -> torch.Tensor:
        """Per-row standardized 11-d element descriptor (avg over co-dopants).

        Used for hard-negative chem-distance. Pulls from
        src.data.element_descriptors._standardized_descriptor_table().
        """
        from src.data.element_descriptors import _standardized_descriptor_table
        table = _standardized_descriptor_table()
        K = len(next(iter(table.values())))
        rows = []
        for lab in dopant_labels:
            lab = str(lab).strip() if lab else "undoped"
            # Co-dopant labels look like "Fe+Sn"; average their descriptors.
            comps = lab.split("+") if "+" in lab else [lab]
            comps = [c.strip() for c in comps if c.strip()]
            if not comps:
                comps = ["undoped"]
            vecs = [table.get(c, table.get("other")) for c in comps]
            arr = torch.tensor(vecs, dtype=torch.float32, device=self.device)
            rows.append(arr.mean(dim=0))
        return torch.stack(rows, dim=0)  # [B, K]

    def _load_pseudo_cache(self) -> None:
        """Lazy-load V53-ζ pseudo-target cache (called once per trainer life)."""
        if self.pseudo_vo_cache is not None or self.self_distill_weight <= 0.0:
            return
        from pathlib import Path
        import numpy as np
        p = Path(self.self_distill_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Self-distill cache missing: {p}. "
                "Run scripts/build_v53z_pseudo_targets.py first."
            )
        arr = np.load(p, allow_pickle=True)
        # Tensors live on CPU; we move per-batch slices to self.device.
        self.pseudo_vo_cache = torch.tensor(arr["pseudo_log_vo"], dtype=torch.float32)
        self.pseudo_std_cache = torch.tensor(arr["pseudo_std"], dtype=torch.float32)

    def _contrastive_pdr_vc_loss(self, model, batch) -> torch.Tensor:
        """Phase 53 V53-ζ + Phase 54 V54-A1: PDR↔V_O latent contrastive loss.

        sincere_mode='doi_label' → vanilla supervised InfoNCE (V53-ζ behavior).
        sincere_mode='chem_sim'  → SINCERE multi-positive over chem buckets.
        sincere_mode='chem_sim+doi' → SINCERE positives restricted to same DOI.

        Hard-negative reweighting (hard_neg_alpha > 0) fires only for
        SINCERE modes; chemistry/process embeddings drive log-weights in
        the negative denominator.
        """
        if self.contrastive_pdr_vc_weight <= 0.0:
            return torch.zeros((), device=self.device)
        proj = getattr(model, "contrastive_proj", None)
        if proj is None:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)
        labels = batch.get("dopant_label", [""] * len(specs))

        group_ids = self._build_group_ids(batch, self.sincere_mode)

        fused = model.get_embedding(graph, specs, process)         # [B, fused_dim]
        z = proj(fused)                                             # [B, contrastive_dim]

        if self.sincere_mode == "doi_label":
            from src.models.contrastive_head import info_nce_loss
            loss = info_nce_loss(
                z, group_ids,
                temperature=self.contrastive_pdr_vc_temperature,
            )
        else:
            from src.models.contrastive_head import sincere_loss, hard_neg_log_weighting
            if self.hard_neg_alpha > 0.0:
                chem_emb = self._build_chem_emb_for_batch(labels)   # [B, 11]
                hnw = hard_neg_log_weighting(chem_emb, process,
                                              alpha=self.hard_neg_alpha)
            else:
                hnw = None
            loss = sincere_loss(
                z, group_ids,
                temperature=self.contrastive_pdr_vc_temperature,
                hard_neg_log_weights=hnw,
            )
        return self.contrastive_pdr_vc_weight * loss

    def _pdr_anchor_sincere_loss(self, model, batch) -> torch.Tensor:
        """V54-A1 PDR-anchor SINCERE arm.

        Anchors are restricted to rows with a non-NaN PDR label; positives and
        negatives are drawn from the full batch via chem-sim grouping. PDR has
        ~226 labels vs ~186 V_O — this arm lets the larger PDR pool drive
        chemistry-axis alignment without enabling a PDR regression head (V44-V8
        confirmed that path is NEG).
        """
        if self.pdr_anchor_weight <= 0.0:
            return torch.zeros((), device=self.device)
        proj = getattr(model, "contrastive_proj", None)
        if proj is None:
            return torch.zeros((), device=self.device)
        if self.sincere_mode == "doi_label":
            # Arm is a SINCERE add-on; meaningless under vanilla InfoNCE keymap.
            return torch.zeros((), device=self.device)

        # Identify PDR-labeled rows.
        target = batch["target"].to(self.device)
        try:
            pdr_idx = self.target_cols.index("photo_dark_ratio")
        except ValueError:
            return torch.zeros((), device=self.device)
        has_pdr = ~torch.isnan(target[:, pdr_idx])
        if has_pdr.sum() < 2:
            return torch.zeros((), device=self.device)

        specs = batch["dopant_spec"]
        labels = batch.get("dopant_label", [""] * len(specs))
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)

        group_ids = self._build_group_ids(batch, self.sincere_mode)

        fused = model.get_embedding(graph, specs, process)         # [B, fused_dim]
        z = proj(fused)                                             # [B, contrastive_dim]

        from src.models.contrastive_head import sincere_loss_per_row, hard_neg_log_weighting
        hnw = None
        if self.hard_neg_alpha > 0.0:
            chem_emb = self._build_chem_emb_for_batch(labels)
            hnw = hard_neg_log_weighting(chem_emb, process, alpha=self.hard_neg_alpha)
        per_row = sincere_loss_per_row(
            z, group_ids,
            temperature=self.contrastive_pdr_vc_temperature,
            hard_neg_log_weights=hnw,
        )
        # Restrict reduction to PDR-labeled anchor rows with non-zero per-row loss
        # (per_row is 0 when a row has no positives — drop it from the mean).
        active = has_pdr & (per_row != 0.0)
        if active.sum() == 0:
            return torch.zeros((), device=self.device)
        return self.pdr_anchor_weight * per_row[active].mean()

    def _charge_vo_distill_loss(self, model, batch) -> torch.Tensor:
        """V54-B1 charge-state V_O soft distill.

        For each batch row, query the frozen charge_vo expert on the
        encoder's pooled crystal embedding to produce (E_f^0, E_f^+1, E_f^+2),
        and MSE-pull the corresponding aux_distill_charge_vo_head(fused).
        Target is standardized per-charge using running batch statistics
        (small-data robust) since the HSE06 E_f_eV values have large abs scale.
        """
        if self.charge_vo_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "charge_vo_expert", None) is None:
            return torch.zeros((), device=self.device)
        if getattr(model, "aux_distill_charge_vo_head", None) is None:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)

        # Pooled crystal embedding from V5 encoder. Use get_embedding to skip
        # the pretrain_head (which would otherwise return 2-d aux predictions).
        crystal_emb = model.encoder.get_embedding(graph)
        if isinstance(crystal_emb, tuple):
            crystal_emb = crystal_emb[0]
        with torch.no_grad():
            target = model.charge_vo_expert(crystal_emb).detach()      # [B, 3]
        # Standardize per-batch (mean/std over the 3 charge states).
        tmean = target.mean()
        tstd = target.std().clamp(min=1.0)
        target_norm = (target - tmean) / tstd

        fused = model.get_embedding(graph, specs, process)
        pred = model.aux_distill_charge_vo_head(fused)                  # [B, 3]
        loss = ((pred - target_norm) ** 2).mean()
        return self.charge_vo_distill_weight * loss

    def _vga_distill_loss(self, model, batch) -> torch.Tensor:
        """V54-B1 V_Ga soft distill on standardized 11-d element descriptors.

        The V_Ga expert is small (no graph input); we feed it the row's
        standardized per-element descriptor (avg over co-dopants if needed).
        """
        if self.vga_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "vga_expert", None) is None:
            return torch.zeros((), device=self.device)
        if getattr(model, "aux_distill_vga_head", None) is None:
            return torch.zeros((), device=self.device)

        labels = batch.get("dopant_label", [""] * len(batch["dopant_spec"]))
        elem_emb = self._build_chem_emb_for_batch([str(l) for l in labels])  # [B, 11]
        with torch.no_grad():
            target = model.vga_expert(elem_emb).detach()              # [B, 3]
        # Already standardized in the expert (normalized at training time).
        # Re-standardize per-batch for robustness.
        tmean = target.mean()
        tstd = target.std().clamp(min=1.0)
        target_norm = (target - tmean) / tstd

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)
        fused = model.get_embedding(graph, specs, process)
        pred = model.aux_distill_vga_head(fused)                       # [B, 3]
        loss = ((pred - target_norm) ** 2).mean()
        return self.vga_distill_weight * loss

    def _dft_scan_distill_loss(self, model, batch) -> torch.Tensor:
        """V57-B1-Quartet+1: pull aux_distill_dft_scan_head(fused) toward the
        frozen 4th expert's prediction of (log[V_O^0], log[V_O^+1], log[V_O^+2])
        derived from (dopant_spec, process). All inputs detach()'d (Rule 1).

        Roadmap caveat #7: start λ=0.05; halt grid if any within-DOI flip.
        """
        if self.dft_scan_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "dft_scan_expert", None) is None:
            return torch.zeros((), device=self.device)
        if getattr(model, "aux_distill_dft_scan_head", None) is None:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)

        # dft_scan_predict internally detaches and returns [B, 3] real-valued
        # log10[V_O^q] for q=0,+1,+2 — same scale as V_O regression head.
        target = model.dft_scan_predict(specs, process)               # [B, 3]
        # Per-batch standardize so different process windows don't dominate.
        tmean = target.mean()
        tstd = target.std().clamp(min=1.0)
        target_norm = (target - tmean) / tstd

        fused = model.get_embedding(graph, specs, process)
        pred = model.aux_distill_dft_scan_head(fused)                 # [B, 3]
        loss = ((pred - target_norm) ** 2).mean()
        return self.dft_scan_distill_weight * loss

    def _mace_latent_distill_loss(self, model, batch) -> torch.Tensor:
        """V57-MACE-Latent: cosine-similarity pull between (fused projected to
        the MACE 64-d space) and the frozen MACE projection for the row's
        primary dopant. Cosine objective is parameter-free at the MACE side
        (frozen projection) so this remains Rule-1 compliant.

        We project fused → 64-d via the same auxiliary head used by the 4th
        expert (re-mapping linearly) and maximize cosine. Equivalent to
        adding a soft InfoNCE positive pair without the full NCE machinery.
        """
        if self.mace_latent_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        if getattr(model, "mace_latent_proj", None) is None:
            return torch.zeros((), device=self.device)

        specs = batch["dopant_spec"]
        mace_z = model.mace_latent_for(specs)        # [B, 64] already detached + L2-normalized
        if mace_z is None:
            return torch.zeros((), device=self.device)
        # Drop rows where the primary dopant isn't in the MACE element vocab
        # (mace_latent_for returns zeros for unknowns — their cosine = 0).
        mask = (mace_z.norm(dim=-1) > 0.5)
        if mask.sum() == 0:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        fused = model.get_embedding(graph, specs, process)            # [B, fused_dim]
        # Project fused → 64-d via a tiny learnable head (created lazily so we
        # don't have to wire it through the model constructor for every config).
        if not hasattr(self, "_mace_aux_head"):
            self._mace_aux_head = nn.Linear(fused.shape[-1], 64).to(self.device)
            # Register in optimizer if not too late; if model already on opt,
            # the head's params will simply not be updated this iter. Acceptable
            # since the cosine signal will still backprop into the encoder.
        import torch.nn.functional as F
        z_pred = self._mace_aux_head(fused)
        z_pred = F.normalize(z_pred, dim=-1)
        cos = (z_pred * mace_z).sum(dim=-1)                           # [B]
        loss = (1.0 - cos)[mask].mean()
        return self.mace_latent_distill_weight * loss

    def _load_pysr_formula(self) -> None:
        """Compile the PySR symbolic formula to a torch callable.

        Reads the formula from `self.pysr_path` (data/processed/pysr_dopant_term.json),
        substitutes the 9 element-descriptor feature names with torch tensor
        slots, and stores a callable that maps [N, 9] → [N]. Called once;
        no-op on subsequent calls.
        """
        if self.pysr_weight <= 0.0 or self.pysr_formula_compiled is not None:
            return
        from pathlib import Path
        import json
        import sympy
        import torch as _torch
        p = Path(self.pysr_path)
        if not p.exists():
            return
        meta = json.loads(p.read_text())
        formula_str = meta.get("best_equation")
        if not formula_str:
            return
        # The formula uses x0..x8 — convert to sympy then to a torch function.
        # x order matches scripts/07_pysr_dopant_term.py:
        #   x0=EN, x1=radius, x2=ox_state, x3=lone_pair, x4=walsh_D,
        #   x5=hardness, x6=chi_mm, x7=r_mm, x8=ox_off
        syms = sympy.symbols([f"x{i}" for i in range(9)])
        try:
            expr = sympy.sympify(formula_str)
            f = sympy.lambdify(syms, expr, modules=[{"exp": _torch.exp,
                                                       "log": _torch.log,
                                                       "Pow": _torch.pow}, "math"])
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"PySR formula compile failed: {exc}; skipping V54-C1 soft loss"
            )
            return
        self.pysr_formula_compiled = f

    def _pysr_dopant_soft_loss(self, model) -> torch.Tensor:
        """V54-C1 soft constraint between ElementHyperNet output and the PySR
        closed-form formula on the same 9-d descriptor space.

        Operates over the KNOWN-element dictionary (CHEM_SIM_GROUP_MAP keys),
        NOT per-batch — the loss is a sum of pairwise MSE across the fixed
        element set. This way the formula constrains the hypernetwork
        regardless of the current batch's element composition.
        """
        if self.pysr_weight <= 0.0:
            return torch.zeros((), device=self.device)
        self._load_pysr_formula()
        if self.pysr_formula_compiled is None:
            return torch.zeros((), device=self.device)
        hn = getattr(model, "element_hypernet", None)
        if hn is None:
            return torch.zeros((), device=self.device)

        from src.data.element_descriptors import (
            _standardized_descriptor_table, _RAW_DESCRIPTORS,
        )
        table = _standardized_descriptor_table()
        # Pick known cations
        elems = [e for e in _RAW_DESCRIPTORS if e not in ("undoped", "other", "H")]
        # Stack standardized descriptors
        desc_std = torch.tensor([table[e] for e in elems],
                                  dtype=torch.float32, device=self.device)  # [N, 11]
        # Hypernet expects [B, K, desc_dim]; treat each elem as B=N, K=1
        descs = desc_std.unsqueeze(1)                                # [N, 1, 11]
        ws = torch.ones(len(elems), 1, device=self.device)            # [N, 1]
        hn_out = hn(descs, ws).squeeze(-1)                           # [N] (out_dim=1)

        # PySR formula uses RAW (un-standardized) descriptors from _RAW_DESCRIPTORS
        raw = torch.tensor([_RAW_DESCRIPTORS[e][:9]
                              if len(_RAW_DESCRIPTORS[e]) >= 9
                              else list(_RAW_DESCRIPTORS[e]) + [0.0]*(9 - len(_RAW_DESCRIPTORS[e]))
                              for e in elems],
                            dtype=torch.float32, device=self.device)
        # raw column order in _RAW_DESCRIPTORS: EN, radius, carrier, Δv, Δperiod,
        # lone_pair, walsh_D, hardness, chi_mm — that's 9 columns, but PySR script
        # used [EN, radius, ox_state, lone_pair, walsh_D, hardness, chi_mm, r_mm, ox_off]
        # which is a DIFFERENT mapping. We must map them to the same order PySR fit on.
        # PySR feature order (from scripts/07_pysr_dopant_term.py:106-108):
        #   x0=EN, x1=radius, x2=ox_state (=Δv+3), x3=lone_pair,
        #   x4=walsh_D, x5=hardness, x6=chi_mm, x7=r_mm, x8=ox_off
        # So we need the corresponding columns from _RAW_DESCRIPTORS:
        #   EN=raw[:,0], radius=raw[:,1], ox_state=raw[:,3]+3, lone_pair=raw[:,5],
        #   walsh_D=raw[:,6], hardness=raw[:,7], chi_mm=raw[:,8], r_mm=raw[:,9], ox_off=raw[:,10]
        raw_full = torch.tensor([_RAW_DESCRIPTORS[e] for e in elems],
                                  dtype=torch.float32, device=self.device)
        x0 = raw_full[:, 0]                  # EN
        x1 = raw_full[:, 1]                  # radius
        x2 = raw_full[:, 3] + 3.0             # ox_state = Δv + 3
        x3 = raw_full[:, 5]                  # lone_pair
        x4 = raw_full[:, 6]                  # walsh_D
        x5 = raw_full[:, 7]                  # hardness
        x6 = raw_full[:, 8]                  # chi_mm
        x7 = raw_full[:, 9]                  # r_mm
        x8 = raw_full[:, 10]                 # ox_off
        try:
            pysr_pred = self.pysr_formula_compiled(x0, x1, x2, x3, x4, x5, x6, x7, x8)
        except Exception:
            return torch.zeros((), device=self.device)
        if not isinstance(pysr_pred, torch.Tensor):
            pysr_pred = torch.tensor(pysr_pred, dtype=torch.float32, device=self.device)
        loss = ((hn_out - pysr_pred.detach()) ** 2).mean()
        return self.pysr_weight * loss

    def _process_axis_infonce_loss(self, model, batch) -> torch.Tensor:
        """V54-A2 process-axis InfoNCE — third contrastive arm.

        Anchor: (dopant_label, DOI) tuple. Positives: rows sharing the same
        anchor — different (T_sub, p_O2, atmosphere) but same chemistry. Pulls
        chemistry latent toward process-invariance.

        Distinct from V53-ζ (which keys on (DOI, dopant_label) too but treats
        all such rows as positives equally — it doesn't distinguish process
        variation as the discriminative axis). Here we restrict to anchors
        that have ≥2 process-distinct positives in batch.
        """
        if self.process_axis_weight <= 0.0:
            return torch.zeros((), device=self.device)
        proj = getattr(model, "contrastive_proj", None)
        if proj is None:
            return torch.zeros((), device=self.device)

        specs = batch["dopant_spec"]
        labels = batch.get("dopant_label", [""] * len(specs))
        dois = batch.get("doi", [""] * len(specs))

        # Build (dopant, DOI) keymap
        keymap: dict[tuple, int] = {}
        ids_list: list[int] = []
        for l, d in zip(labels, dois):
            key = (str(l), str(d))
            if key not in keymap:
                keymap[key] = len(keymap)
            ids_list.append(keymap[key])
        group_ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        # Require ≥2 rows per group; mark singletons as sentinel (-1)
        counts = torch.bincount(group_ids.clamp(min=0))
        singleton_mask = counts[group_ids] < 2
        group_ids = group_ids.masked_fill(singleton_mask, -1)
        if (group_ids != -1).sum() < 2:
            return torch.zeros((), device=self.device)

        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        fused = model.get_embedding(graph, specs, process)         # [B, fused_dim]
        z = proj(fused)                                             # [B, contrastive_dim]

        from src.models.contrastive_head import sincere_loss
        loss = sincere_loss(z, group_ids,
                            temperature=self.process_axis_temperature)
        return self.process_axis_weight * loss

    def _pdr_self_distill_loss(self, model, batch) -> torch.Tensor:
        """V54-A1 self-distill from V53-ζ pseudo-targets on V_O-unlabeled rows.

        Reads pseudo_log_vo + pseudo_std from data/processed/v53z_pseudo_targets_vo.npz
        (keyed by sample_idx in the post-filter dataset). Targets enter via
        aux_self_distill_head(fused) → MSE in standardized log10[V_O] space,
        optionally weighted by inverse pseudo_std (high-confidence V53-ζ rows
        contribute more).
        """
        if self.self_distill_weight <= 0.0:
            return torch.zeros((), device=self.device)
        head = getattr(model, "aux_self_distill_head", None)
        if head is None:
            return torch.zeros((), device=self.device)
        self._load_pseudo_cache()
        if self.pseudo_vo_cache is None:
            return torch.zeros((), device=self.device)

        sample_idx = batch.get("sample_idx", None)
        if sample_idx is None:
            return torch.zeros((), device=self.device)
        sample_idx = sample_idx.to(torch.long)
        valid_idx_mask = sample_idx.ge(0) & sample_idx.lt(self.pseudo_vo_cache.shape[0])
        if valid_idx_mask.sum() == 0:
            return torch.zeros((), device=self.device)

        # Gather pseudo & std (on CPU cache → device).
        safe_idx = sample_idx.clamp(min=0, max=self.pseudo_vo_cache.shape[0] - 1)
        pseudo_cpu = self.pseudo_vo_cache[safe_idx]                # [B]
        pseudo_std_cpu = self.pseudo_std_cache[safe_idx]
        pseudo = pseudo_cpu.to(self.device)
        pseudo_std = pseudo_std_cpu.to(self.device)
        valid = valid_idx_mask.to(self.device) & ~torch.isnan(pseudo)
        if valid.sum() == 0:
            return torch.zeros((), device=self.device)

        # Standardize using current fold target stats (set elsewhere); fall back
        # to raw log10 space if stats are unavailable.
        tmean = getattr(self, "_cur_target_mean", None)
        tstd = getattr(self, "_cur_target_std", None)
        try:
            vc_idx = self.target_cols.index("vacancy_concentration")
        except ValueError:
            vc_idx = None
        if tmean is not None and tstd is not None and vc_idx is not None:
            mean = tmean[vc_idx] if hasattr(tmean, "__getitem__") else tmean
            std = tstd[vc_idx] if hasattr(tstd, "__getitem__") else tstd
            mean_t = torch.as_tensor(mean, dtype=pseudo.dtype, device=self.device)
            std_t = torch.as_tensor(std, dtype=pseudo.dtype, device=self.device).clamp(min=1e-3)
            pseudo_norm = (pseudo - mean_t) / std_t
        else:
            pseudo_norm = pseudo

        graph = batch["graph"].to(self.device)
        specs = batch["dopant_spec"]
        process = batch["process"].to(self.device)
        fused = model.get_embedding(graph, specs, process)         # [B, fused_dim]
        pred = head(fused).squeeze(-1)                              # [B]

        sq_err = (pred[valid] - pseudo_norm[valid].detach()) ** 2
        if self.self_distill_inv_var:
            # weight = 1 / (std + 0.1)^2; high-std rows down-weighted; normalize so mean weight ≈ 1
            w = 1.0 / (pseudo_std[valid].clamp(min=0.1) ** 2)
            w = w / w.mean().clamp(min=1e-6)
            loss = (w * sq_err).mean()
        else:
            loss = sq_err.mean()
        return self.self_distill_weight * loss

    def _srh_coupling_loss(self, model, batch, pred) -> torch.Tensor:
        """Phase 44 V8: SRH coupling penalty between V_O and PDR predictions.

        Physics: oxygen vacancies are deep traps that increase SRH
        recombination, which suppresses photo-current and therefore PDR.
        The "high V_O AND high PDR" corner is therefore unphysical.

        Implementation: a soft hinge penalty
            L_srh = mean_i [ relu(pred_VC_std[i] - τ_VC) *
                             relu(pred_PDR_std[i] - τ_PDR) ]^2
        on rows where both targets are valid (i.e. both heads contributed
        non-trivial predictions).

        The hinge fires ONLY in the unphysical corner — for samples below
        either threshold, gradient is zero. This means the SRH coupling
        cannot create slope sign flips (which require gradient on the bulk
        of the data, not just a corner). It can only push back from the
        unphysical corner. Symmetric in VC and PDR.

        Predictions are in standardized space (target_standardize=true);
        τ defaults to 1.0 ≈ 1σ above mean.
        """
        if self.srh_coupling_weight <= 0.0:
            return torch.zeros((), device=self.device)

        # `pred` is the model output in standardized space; shape [B, T] where
        # T = len(self.target_cols). Need both VC and PDR target indices.
        target_cols = self.target_cols
        if "vacancy_concentration" not in target_cols:
            return torch.zeros((), device=self.device)
        if "photo_dark_ratio" not in target_cols:
            return torch.zeros((), device=self.device)
        idx_vc  = target_cols.index("vacancy_concentration")
        idx_pdr = target_cols.index("photo_dark_ratio")

        target = batch["target"].to(self.device)
        # Active row mask: both targets valid
        valid = ~torch.isnan(target).any(dim=-1)
        if valid.sum() < 1:
            return torch.zeros((), device=self.device)

        p_vc  = pred[valid, idx_vc]
        p_pdr = pred[valid, idx_pdr]

        hinge = (
            torch.relu(p_vc  - self.srh_tau_vc)
            * torch.relu(p_pdr - self.srh_tau_pdr)
        )
        loss = (hinge ** 2).mean()
        return self.srh_coupling_weight * loss

    def _cross_element_ranking_loss(self, model, batch) -> torch.Tensor:
        """Phase 23: BPR-style cross-element ranking loss.

        Within batch, for pairs (i, j) with different cation elements where
        both true targets are non-NaN AND |true_diff| > min_dtrue, encourage
        sign(pred_i - pred_j) == sign(true_i - true_j) via:
            loss = -log(sigmoid((pred_i - pred_j) * sign(true_i - true_j)))

        Targets the cross-element ranking failure (e.g., PDR ρ=0 on sputter
        where 16A model can't rank Mg > Sn correctly). Doesn't force absolute
        directions, only relative ordering — minimal conflict with existing
        priors.
        """
        if self.cross_elem_rank_weight <= 0.0:
            return torch.zeros((), device=self.device)
        from src.data.dopant_spec import DopantSpec
        import torch.nn.functional as F

        specs = batch["dopant_spec"]
        target_full = batch["target"].to(self.device)  # [B, n_targets]
        k = self.cross_elem_rank_target_idx
        if target_full.dim() < 2 or target_full.shape[1] <= k:
            return torch.zeros((), device=self.device)
        target_k = target_full[:, k]                     # [B]

        # Element label per sample (majority cation)
        elems = []
        for s in specs:
            sp = DopantSpec.parse(s)
            if sp.is_undoped or not sp.components:
                elems.append("__undoped__")
            else:
                elems.append(max(sp.components, key=lambda c: c.conc).cation)

        # Forward pass for predictions
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        emb = model.get_embedding(graph, specs, process)
        pred = model._heads_from_fused(emb, dopant_specs=specs, process=process)
        if pred.dim() < 2 or pred.shape[1] <= k:
            return torch.zeros((), device=self.device)
        pred_k = pred[:, k]                              # [B]

        n = len(specs)
        bpr_terms = []
        for i in range(n):
            t_i = target_k[i]
            if torch.isnan(t_i): continue
            for j in range(i + 1, n):
                t_j = target_k[j]
                if torch.isnan(t_j): continue
                if elems[i] == elems[j]: continue
                true_diff = (t_i - t_j).item()
                if abs(true_diff) < self.cross_elem_rank_min_dtrue: continue
                sign = 1.0 if true_diff > 0 else -1.0
                pred_diff = pred_k[i] - pred_k[j]
                # BPR log-sigmoid: maximise log σ(sign·(pred_i - pred_j))
                bpr_terms.append(-F.logsigmoid(sign * pred_diff))

        if not bpr_terms:
            return torch.zeros((), device=self.device)
        return self.cross_elem_rank_weight * torch.stack(bpr_terms).mean()

    # ── Phase 25-soft: physics envelope as hinge ceiling ─────────────────────
    def _physics_envelope_loss_pdr(
        self,
        model,
        batch,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge loss penalising predictions that exceed the physics-allowed
        ceiling env(λ, T) · (log_PDR_max − log_PDR_floor) + log_PDR_floor.

        env_λ = sigmoid((E_photon − E_g + δ)/σ_λ), env_T = exp(-(T−T_opt)²/2σ_T²),
        env = env_λ · env_T ∈ [0, 1].

        Sub-bandgap or far-from-T_opt → env→0 → ceiling collapses to floor
        (log_PDR=0, no photoresponse). The model is free to predict below this
        ceiling but is penalised for exceeding it. Envelope module
        (model.physics_envelope_pdr) supplies trainable physics constants whose
        gradients flow back through this hinge.
        """
        weight = self.physics_envelope_pdr_loss_weight
        if weight <= 0.0:
            return torch.zeros((), device=self.device)
        envelope = getattr(model, "physics_envelope_pdr", None)
        if envelope is None:
            return torch.zeros((), device=self.device)

        import torch.nn.functional as F
        from src.models.physics_features import compute_physics_features

        process = batch["process"].to(self.device)
        dopant_specs = batch["dopant_spec"]
        physics = compute_physics_features(dopant_specs, process)
        env = envelope.envelope(process, physics)            # [B] in [0, 1]

        k = self.physics_envelope_pdr_target_idx
        pred_k = pred[:, k]                                  # standardized

        # Ceiling in physical (log_PDR) space, then standardize to match `pred`.
        ceiling_log = (
            env * self.physics_envelope_pdr_ceiling
            + (1.0 - env) * self.physics_envelope_pdr_floor_log10
        )
        tmean = getattr(self, "_phys_target_mean", None)
        tstd = getattr(self, "_phys_target_std", None)
        if tmean is not None and tstd is not None:
            ceiling_std = (ceiling_log - tmean) / tstd
        else:
            ceiling_std = ceiling_log

        hinge = F.relu(pred_k - ceiling_std).pow(2)
        return weight * hinge.mean()

    # ── Phase 6A: chemistry-family cos-similarity regularizer ────────────────
    def _family_cos_loss(self, model) -> torch.Tensor:
        """Pull same-family DopantStream embeddings together via 1-cos penalty.

        Returns 0 (as a tensor on the correct device) if the model has no
        DopantStream or if ``family_cos_weight`` is 0. Skips the "other"
        bucket so miscellaneous elements aren't forced together.
        """
        if self.family_cos_weight <= 0.0 or model.dopant_stream is None:
            return torch.zeros((), device=self.device)
        emb = model.dopant_stream.element_embedding.weight  # [n_dopants, d]
        fam = model.dopant_stream.family_groups            # [n_dopants]
        from src.data.element_descriptors import FAMILY_OTHER_IDX
        in_real_family = fam < FAMILY_OTHER_IDX
        if in_real_family.sum() < 2:
            return torch.zeros((), device=emb.device)
        emb_n = torch.nn.functional.normalize(emb, dim=-1)
        cos = emb_n @ emb_n.t()                             # [n, n]
        same = (fam.unsqueeze(0) == fam.unsqueeze(1))
        mask = same & in_real_family.unsqueeze(0) & in_real_family.unsqueeze(1)
        mask = mask & ~torch.eye(len(fam), dtype=torch.bool, device=emb.device)
        n_pairs = mask.sum().clamp(min=1)
        loss = ((1.0 - cos) * mask.float()).sum() / n_pairs
        return self.family_cos_weight * loss

    # ── Public entry point ────────────────────────────────────────────────────

    def run_cv(
        self,
        random_state: int = 42,
        groups: np.ndarray | None = None,
    ) -> dict:
        """
        Run k-fold CV.

        If ``groups`` is provided (e.g. one paper DOI per sample), splits use
        :class:`sklearn.model_selection.GroupKFold` so no group spans both
        train and val — this catches paper-level leakage that StratifiedKFold
        misses when multiple rows come from the same paper.

        Otherwise falls back to StratifiedKFold on cation_label (e.g. "Fe",
        "Fe+Sn", "Sn+Mg") so each fold sees as many distinct dopant types as
        possible.

        Args:
            random_state: Seed for shuffle. Varied for ensemble.
            groups: Optional [N] array of group keys (e.g. doi strings).
                    When set, triggers GroupKFold mode.

        Returns:
            dict with per-target mean/std MAE, RMSE, R² across folds,
            and XGBoost results if enabled.
        """
        all_labels = np.array(self.dataset.get_dopant_labels())
        indices = np.arange(len(self.dataset))
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != len(indices):
                raise ValueError(
                    f"groups length {groups.shape[0]} != dataset size {len(indices)}"
                )
            # GroupKFold is deterministic in sklearn <1.6; shuffle groups by
            # their first-occurrence index to honour random_state for seed
            # diversity in ensembles.
            unique_groups = pd.unique(groups) if hasattr(pd := __import__("pandas"), "unique") else np.unique(groups)
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(len(unique_groups))
            remap = {g: int(perm[i]) for i, g in enumerate(unique_groups)}
            shuffled_groups = np.array([remap[g] for g in groups])
            skf = GroupKFold(n_splits=self.cv_folds)
            split_iter = skf.split(indices, all_labels, groups=shuffled_groups)
            logger.info(
                f"GroupKFold: {len(unique_groups)} unique groups → "
                f"{self.cv_folds} folds (random_state={random_state} shuffles group order)"
            )
        else:
            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=random_state
            )
            split_iter = skf.split(indices, all_labels)

        mlp_fold_metrics: dict[str, list] = defaultdict(list)
        xgb_fold_metrics: dict[str, list] = defaultdict(list)
        gp_fold_metrics:  dict[str, list] = defaultdict(list)
        stop_epochs: list[int] = []
        oof_rows: list[dict] = []

        for fold, (train_idx, val_idx) in enumerate(split_iter):
            logger.info(
                f"=== Stage 3 Fold {fold+1}/{self.cv_folds} "
                f"(train={len(train_idx)}, val={len(val_idx)}) ==="
            )
            # Phase 54 V54-A2: reset SNGP RFF precision matrix at fold start
            # to avoid cross-fold leakage (each fold builds Σ from scratch).
            # No-op when model doesn't have a SNGP head.
            if hasattr(self.initial_model, "reset_sngp_precision"):
                try:
                    self.initial_model.reset_sngp_precision()
                except Exception:
                    pass
            logger.info(
                f"  Train dopants: {sorted(set(all_labels[train_idx]))}"
            )
            logger.info(
                f"  Val dopants:   {sorted(set(all_labels[val_idx]))}"
            )

            mlp_metrics, xgb_metrics, gp_metrics, _, best_epoch, fold_oof = self._run_fold(
                fold, train_idx, val_idx
            )
            stop_epochs.append(best_epoch)
            oof_rows.extend(fold_oof)

            for k, v in mlp_metrics.items():
                mlp_fold_metrics[k].append(v)
            for k, v in xgb_metrics.items():
                xgb_fold_metrics[k].append(v)
            for k, v in gp_metrics.items():
                gp_fold_metrics[k].append(v)

        # Store median stop epoch for use by train_final_model()
        self._median_stop_epoch = int(np.median(stop_epochs))
        logger.info(
            f"  Fold stop epochs: {stop_epochs} "
            f"→ median={self._median_stop_epoch}"
        )

        # Aggregate
        mlp_summary = self._aggregate(mlp_fold_metrics, prefix="mlp")
        xgb_summary = self._aggregate(xgb_fold_metrics, prefix="xgb")
        gp_summary  = self._aggregate(gp_fold_metrics,  prefix="gp")
        summary = {**mlp_summary, **xgb_summary, **gp_summary}

        # ── Platt scaling (post-hoc slope correction) ────────────────────────
        self._platt_params = self._fit_platt_scaling(oof_rows)
        for row in oof_rows:
            for col in self.target_cols:
                a, b = self._platt_params[col]
                raw = row.get(f"{col}_pred", float("nan"))
                if not np.isnan(raw):
                    row[f"{col}_pred_platt"] = a * raw + b

        logger.info("\n" + "="*60)
        logger.info(f"Stage 3 {self.cv_folds}-Fold CV Summary")
        logger.info("="*60)
        for k, v in summary.items():
            logger.info(f"  {k}: {v:.4f}")

        return summary, oof_rows

    def train_final_model(self, n_epochs: int | None = None) -> Ga2O3Net:
        """
        Train a final model on ALL data (no validation split).

        Called after run_cv() to produce a deployment-ready model.
        Uses the median early-stop epoch from CV as the training budget.

        Args:
            n_epochs: Number of training epochs.  If None, uses the median
                      stop epoch recorded during run_cv().

        Returns:
            Fully trained Ga2O3Net with scalers fitted on all data.
        """
        n_epochs = n_epochs or getattr(self, "_median_stop_epoch", self.epochs)
        all_indices = np.arange(len(self.dataset))
        all_specs = self.dataset.get_dopant_specs()
        all_process = self.dataset.get_process_array()
        all_labels = np.array(self.dataset.get_dopant_labels())

        logger.info(
            f"Training final model on {len(all_indices)} samples "
            f"for {n_epochs} epochs..."
        )

        le = LabelEncoder().fit(all_labels.tolist())

        model = copy.deepcopy(self.initial_model).to(self.device)
        model.process_stream.fit_scaler(all_process)
        model.composition_stream.fit_scaler(dopant_specs=all_specs)

        # Encoder freeze / partial unfreeze
        if self.encoder_unfreeze_last_n > 0:
            model.encoder.unfreeze_last_n(self.encoder_unfreeze_last_n)
        else:
            model.freeze_encoder()

        # Initialise composition decoder if reconstruction loss is enabled
        if self.recon_weight > 0.0:
            model.init_composition_decoder()

        aug_dataset = self._make_aug_dataset()
        aug_dataset._graph_cache = self.dataset._graph_cache
        aug_dataset._specs = self.dataset._specs

        loader = DataLoader(
            aug_dataset,
            batch_size=min(self.batch_size, len(all_indices)),
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Target standardization for final model (full-dataset stats)
        if self.target_standardize:
            full_mean, full_std = self._compute_target_stats(all_indices)
            self._final_target_mean = full_mean
            self._final_target_std  = full_std
            _tmean_f = torch.tensor(full_mean, device=self.device)
            _tstd_f  = torch.tensor(full_std,  device=self.device)
            logger.info(
                f"  Final model target z-score: mean={full_mean}, std={full_std}"
            )
        else:
            self._final_target_mean = None
            self._final_target_std  = None
            _tmean_f, _tstd_f = None, None

        # Phase 25-soft: same standardization plumbing as in CV folds
        if self.physics_envelope_pdr_loss_weight > 0.0 and _tmean_f is not None:
            k = self.physics_envelope_pdr_target_idx
            self._phys_target_mean = _tmean_f[k] if _tmean_f.ndim > 0 else _tmean_f
            self._phys_target_std  = _tstd_f[k]  if _tstd_f.ndim  > 0 else _tstd_f
        else:
            self._phys_target_mean = None
            self._phys_target_std  = None

        # Phase 54 V54-A1: expose full per-target stats so _pdr_self_distill_loss
        # can z-score V53-ζ pseudo-targets into the same space as the regression.
        self._cur_target_mean = _tmean_f
        self._cur_target_std  = _tstd_f
        self._set_dft_anchor_stats(model, _tmean_f, _tstd_f)  # V58 M9 hook

        self.loss_fn = copy.deepcopy(self._initial_loss_fn).to(self.device)
        optimizer = self._build_optimizer(model)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=self.base_lr * 0.01
        )

        for epoch in range(1, n_epochs + 1):
            self._cur_epoch = epoch; self._cur_total_epochs = n_epochs  # Phase 59 DCC warm-up
            model.train()
            for batch in loader:
                graph = batch["graph"].to(self.device)
                process = batch["process"].to(self.device)
                target = batch["target"].to(self.device)
                dopant_specs = batch["dopant_spec"]
                dopant_labels = batch["dopant_label"]

                # Target standardization
                if _tmean_f is not None:
                    target = self._standardize_target(target, _tmean_f, _tstd_f)

                label_classes = self._encode_labels(le, dopant_labels)

                optimizer.zero_grad()
                emb = model.get_embedding(graph, dopant_specs, process)

                # Reconstruction auxiliary loss — all samples, no labels needed
                recon_pred, recon_target = self._get_recon(model, emb, dopant_specs)

                sample_w = batch.get("sample_weight")
                if sample_w is not None:
                    sample_w = sample_w.to(self.device)

                if self.mixup_alpha > 0.0:
                    emb_in, target_in = self._mixup_batch(
                        emb.detach(), target, self.mixup_alpha,
                        label_kernel_bw=self.mixup_label_kernel_bw,
                        primary_target_idx=self.mixup_primary_target_idx,
                    )
                    pred = model._heads_from_fused(emb_in, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target_in, emb_in, label_classes, model,
                        recon_pred=recon_pred, recon_target=recon_target,
                    )
                else:
                    pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target, emb, label_classes, model,
                        recon_pred=recon_pred, recon_target=recon_target,
                        sample_weights=sample_w,
                    )

                # Skip batch if loss has no gradient (all targets NaN, no l2)
                if not loss.requires_grad:
                    continue
                loss = loss + self._family_cos_loss(model)   # Phase 6A
                loss = loss + self._monotonic_constraint_loss(model, batch)   # Phase 6C
                loss = loss + self._monotonic_vc_concentration_loss(model, batch)  # Phase 17
                loss = loss + self._monotonic_vc_within_doi_loss(model, batch)  # Phase 20
                loss = loss + self._atmosphere_ordinal_vc_loss(model, batch)  # Phase 32
                loss = loss + self._atmosphere_ordinal_pdr_loss(model, batch)  # Phase 28-PDR
                loss = loss + self._anneal_temperature_sign_loss(model, batch)  # Phase 36
                loss = loss + self._element_class_contrastive_loss(model, batch)  # Phase 35
                loss = loss + self._cross_element_ranking_loss(model, batch)  # Phase 23
                loss = loss + self._cross_method_dopant_pull_loss(model, batch)  # Phase 42
                loss = loss + self._encoder_distill_loss(model, batch)  # Phase 43
                loss = loss + self._contrastive_pdr_vc_loss(model, batch)  # Phase 53 V53-ζ
                loss = loss + self._pdr_anchor_sincere_loss(model, batch)   # Phase 54 V54-A1
                loss = loss + self._pdr_self_distill_loss(model, batch)     # Phase 54 V54-A1
                loss = loss + self._process_axis_infonce_loss(model, batch) # Phase 54 V54-A2
                loss = loss + self._pysr_dopant_soft_loss(model)            # Phase 54 V54-C1
                loss = loss + self._charge_vo_distill_loss(model, batch)    # Phase 54 V54-B1
                loss = loss + self._vga_distill_loss(model, batch)          # Phase 54 V54-B1
                loss = loss + self._dft_scan_distill_loss(model, batch)      # V57-B1-Quartet+1
                loss = loss + self._mace_latent_distill_loss(model, batch)   # V57-MACE-Latent
                loss = loss + self._dft_residual_reg_loss(model, batch)       # V58
                loss = loss + self._tier1c_dft_consistency_loss(model, batch) # V58
                loss = loss + self._kroger_distill_loss(model, batch)  # Phase 53 V53-α
                loss = loss + self._srh_coupling_loss(model, batch, pred)  # Phase 44 V8
                loss = loss + self._dcc_loss(model, batch)  # Phase 59 V59-CALM DCC
                loss = loss + self._physics_envelope_loss_pdr(model, batch, pred)  # Phase 25-soft
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                optimizer.step()

            scheduler.step()

            if epoch % 100 == 0 or epoch == n_epochs:
                logger.info(
                    f"  Final model epoch {epoch}/{n_epochs} "
                    f"loss={loss.item():.4f}"
                )

        model.eval()
        logger.info("Final model training complete.")
        return model

    # ── Single fold ───────────────────────────────────────────────────────────

    # ── V58 helpers ───────────────────────────────────────────────────────────
    # Tier-1C unsupervised / poorly-covered elements (doc §1.1) — these are the
    # OFFSET/ambiguous cases the DFT anchor is meant to rescue.
    _TIER1C_ELEMENTS = {"Sb", "F", "Cu", "W", "Bi", "Er", "Eu", "B", "V"}

    def _set_dft_anchor_stats(self, model, tmean, tstd) -> None:
        """Give DFTAnchoredBrouwerHead (M9) the VC target's physical (log10) z-score
        stats so its absolute DFT anchor maps into standardized space. No-op otherwise."""
        if tmean is None or not getattr(model, "dft_stream_enabled", False):
            return
        if not hasattr(model, "set_dft_target_stats"):
            return
        if "vacancy_concentration" not in self.target_cols:
            return
        vc = self.target_cols.index("vacancy_concentration")
        m = tmean[vc] if getattr(tmean, "ndim", 0) > 0 else tmean
        s = tstd[vc] if getattr(tstd, "ndim", 0) > 0 else tstd
        model.set_dft_target_stats(float(m), float(s))

    def _v58_vc_head(self, model):
        from src.models.brouwer_head_dft import DFTAnchoredBrouwerHead
        if not getattr(model, "dft_stream_enabled", False):
            return None
        h = model.heads["vacancy_concentration"] if "vacancy_concentration" in model.heads else None
        return h if isinstance(h, DFTAnchoredBrouwerHead) else None

    def _dft_residual_reg_loss(self, model, batch) -> torch.Tensor:
        """L2 on the bounded δ residual of M9 over DFT-valid rows (doc §4.5, w=0.05)."""
        if self.dft_residual_reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        h = self._v58_vc_head(model)
        if h is None:
            return torch.tensor(0.0, device=self.device)
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        specs = batch["dopant_spec"]
        fused = model.get_embedding(graph, specs, process)
        delta = torch.tanh(h.residual_net(fused)) * h.max_residual    # [B,1] dex
        dft_features = model._lookup_dft_features(specs, process)
        valid = model.dft_stream.valid_mask(dft_features)
        if valid.any():
            delta = delta.squeeze(-1)[valid]
            return self.dft_residual_reg_weight * (delta ** 2).mean()
        return torch.tensor(0.0, device=self.device)

    def _tier1c_dft_consistency_loss(self, model, batch) -> torch.Tensor:
        """Pull Tier-1C unsupervised elements toward the pure DFT anchor (doc §4.5,
        w=0.1) — the §1.1 OFFSET fix for Sb/F/Cu/W/Bi/Er/Eu/B/V."""
        if self.tier1c_dft_consistency_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        h = self._v58_vc_head(model)
        if h is None:
            return torch.tensor(0.0, device=self.device)
        from src.data.dopant_spec import DopantSpec
        graph = batch["graph"].to(self.device)
        process = batch["process"].to(self.device)
        specs = batch["dopant_spec"]
        # mask: Tier-1C element AND a valid DFT anchor
        dft_features = model._lookup_dft_features(specs, process)
        valid = model.dft_stream.valid_mask(dft_features)
        is_t1c = torch.zeros(len(specs), dtype=torch.bool, device=self.device)
        for i, s in enumerate(specs):
            try:
                sp = DopantSpec.parse(s.strip())
                cats = {c.cation for c in sp.components} if sp and sp.components else set()
            except Exception:
                cats = set()
            if cats & self._TIER1C_ELEMENTS:
                is_t1c[i] = True
        mask = valid & is_t1c
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        fused = model.get_embedding(graph, specs, process)
        pred_vc = model._heads_from_fused(fused, dopant_specs=specs, process=process)
        vc = self.target_cols.index("vacancy_concentration")
        pred_vc = pred_vc[:, vc]                                   # standardized
        anchor = h.dft_anchor_only(dft_features, process).squeeze(-1)  # standardized, no δ
        diff = (pred_vc - anchor)[mask]
        return self.tier1c_dft_consistency_weight * (diff ** 2).mean()

    def _run_fold(
        self,
        fold: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Train and evaluate one CV fold.

        Steps:
          1. Fit scalers on train_idx only (no leakage).
          2. Deep-copy initial model; freeze encoder if configured.
          3. Train MLP head with early stopping.
          4. Evaluate MLP on val_idx.
          5. If xgboost: extract embeddings for train+val, train XGB, evaluate.

        Returns:
            (mlp_metrics, xgb_metrics) — dicts of {target_metric: value}.
        """
        # ── 1. Fit scalers on train fold only ─────────────────────────────────
        all_specs = self.dataset.get_dopant_specs()
        all_process = self.dataset.get_process_array()

        train_specs = [all_specs[i] for i in train_idx]
        train_process = all_process[train_idx]

        # LabelEncoder on train labels (val labels must already exist in train
        # because of stratification, but we handle unseen gracefully below)
        all_labels = np.array(self.dataset.get_dopant_labels())
        train_labels = all_labels[train_idx].tolist()
        le = LabelEncoder()
        le.fit(train_labels)

        # Deep copy model; fit scalers on train data
        model = copy.deepcopy(self.initial_model).to(self.device)
        model.process_stream.fit_scaler(train_process)
        model.composition_stream.fit_scaler(dopant_specs=train_specs)

        # Encoder freeze / partial unfreeze
        if self.encoder_unfreeze_last_n > 0:
            model.encoder.unfreeze_last_n(self.encoder_unfreeze_last_n)
        else:
            model.freeze_encoder()

        # Initialise composition decoder if reconstruction loss is enabled
        if self.recon_weight > 0.0:
            model.init_composition_decoder()

        # ── 1b. Target standardization (per-fold, train-only stats) ──────────
        if self.target_standardize:
            fold_target_mean, fold_target_std = self._compute_target_stats(train_idx)
            _tmean = torch.tensor(fold_target_mean, device=self.device)
            _tstd  = torch.tensor(fold_target_std,  device=self.device)
            logger.info(
                f"  Target z-score: mean={fold_target_mean}, std={fold_target_std}"
            )
        else:
            fold_target_mean, fold_target_std = None, None
            _tmean, _tstd = None, None

        # Phase 25-soft: expose the PDR-target standardization so
        # _physics_envelope_loss_pdr can map ceiling_log10 into standardized space.
        if self.physics_envelope_pdr_loss_weight > 0.0 and _tmean is not None:
            k = self.physics_envelope_pdr_target_idx
            self._phys_target_mean = _tmean[k] if _tmean.ndim > 0 else _tmean
            self._phys_target_std  = _tstd[k]  if _tstd.ndim  > 0 else _tstd
        else:
            self._phys_target_mean = None
            self._phys_target_std  = None

        # Phase 54 V54-A1: full per-target stats for _pdr_self_distill_loss.
        self._cur_target_mean = _tmean
        self._cur_target_std  = _tstd
        self._set_dft_anchor_stats(model, _tmean, _tstd)  # V58 M9 hook (per fold)

        # ── 2. Build loaders ──────────────────────────────────────────────────
        # Augmented training subset (shares graph cache)
        aug_dataset = self._make_aug_dataset()
        aug_dataset._graph_cache = self.dataset._graph_cache
        aug_dataset._specs = self.dataset._specs

        train_loader = DataLoader(
            Subset(aug_dataset, train_idx.tolist()),
            batch_size=min(self.batch_size, len(train_idx)),
            shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_idx.tolist()),
            batch_size=len(val_idx),
            shuffle=False, collate_fn=collate_fn,
        )

        # Phase 12A: collect train fold's DOIs for synth bank filtering
        if self.synth_bank is not None:
            try:
                train_dois = set(
                    self.dataset.df.iloc[train_idx]["doi"]
                    .fillna("—").astype(str).unique().tolist()
                )
                self._cur_train_dois = train_dois
                n_eligible = self.synth_bank.n_eligible_for_dois(train_dois)
                logger.info(
                    f"  Latent synth bank: {n_eligible} synth rows eligible "
                    f"({len(train_dois)} train DOIs)"
                )
            except Exception as e:
                logger.warning(f"  Could not derive train DOIs: {e}; using all synth")
                self._cur_train_dois = None

        # ── 3. Train MLP head ─────────────────────────────────────────────────
        # Reset loss_fn per fold so log_vars don't bleed across folds
        self.loss_fn = copy.deepcopy(self._initial_loss_fn).to(self.device)

        optimizer = self._build_optimizer(model)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.base_lr * 0.01
        )

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = 1
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self._cur_epoch = epoch; self._cur_total_epochs = self.epochs  # Phase 59 DCC warm-up
            model.train()
            for batch in train_loader:
                graph = batch["graph"].to(self.device)
                process = batch["process"].to(self.device)
                target = batch["target"].to(self.device)
                dopant_specs = batch["dopant_spec"]
                dopant_labels = batch["dopant_label"]

                # Target standardization (NaN stays NaN automatically)
                if _tmean is not None:
                    target = self._standardize_target(target, _tmean, _tstd)

                # Map labels to int (handle any unseen val label gracefully)
                label_classes = self._encode_labels(le, dopant_labels)

                optimizer.zero_grad()
                emb = model.get_embedding(graph, dopant_specs, process)

                # Reconstruction auxiliary loss — all samples, no labels needed
                recon_pred, recon_target = self._get_recon(model, emb, dopant_specs)

                sample_w = batch.get("sample_weight")
                if sample_w is not None:
                    sample_w = sample_w.to(self.device)

                # Embedding-space Mixup (optional)
                if self.mixup_alpha > 0.0:
                    emb_in, target_in = self._mixup_batch(
                        emb.detach(), target, self.mixup_alpha,
                        label_kernel_bw=self.mixup_label_kernel_bw,
                        primary_target_idx=self.mixup_primary_target_idx,
                    )
                    pred = model._heads_from_fused(emb_in, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target_in, emb_in, label_classes, model,
                        recon_pred=recon_pred, recon_target=recon_target,
                    )
                else:
                    pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target, emb, label_classes, model,
                        recon_pred=recon_pred, recon_target=recon_target,
                        sample_weights=sample_w,
                    )

                # Skip batch if loss has no gradient (all targets NaN, no l2)
                if not loss.requires_grad:
                    continue
                loss = loss + self._family_cos_loss(model)   # Phase 6A
                loss = loss + self._monotonic_constraint_loss(model, batch)   # Phase 6C
                loss = loss + self._monotonic_vc_concentration_loss(model, batch)  # Phase 17
                loss = loss + self._monotonic_vc_within_doi_loss(model, batch)  # Phase 20
                loss = loss + self._atmosphere_ordinal_vc_loss(model, batch)  # Phase 32
                loss = loss + self._atmosphere_ordinal_pdr_loss(model, batch)  # Phase 28-PDR
                loss = loss + self._anneal_temperature_sign_loss(model, batch)  # Phase 36
                loss = loss + self._element_class_contrastive_loss(model, batch)  # Phase 35
                loss = loss + self._cross_element_ranking_loss(model, batch)  # Phase 23
                loss = loss + self._cross_method_dopant_pull_loss(model, batch)  # Phase 42
                loss = loss + self._encoder_distill_loss(model, batch)  # Phase 43
                loss = loss + self._contrastive_pdr_vc_loss(model, batch)  # Phase 53 V53-ζ
                loss = loss + self._pdr_anchor_sincere_loss(model, batch)   # Phase 54 V54-A1
                loss = loss + self._pdr_self_distill_loss(model, batch)     # Phase 54 V54-A1
                loss = loss + self._process_axis_infonce_loss(model, batch) # Phase 54 V54-A2
                loss = loss + self._pysr_dopant_soft_loss(model)            # Phase 54 V54-C1
                loss = loss + self._charge_vo_distill_loss(model, batch)    # Phase 54 V54-B1
                loss = loss + self._vga_distill_loss(model, batch)          # Phase 54 V54-B1
                loss = loss + self._dft_scan_distill_loss(model, batch)      # V57-B1-Quartet+1
                loss = loss + self._mace_latent_distill_loss(model, batch)   # V57-MACE-Latent
                loss = loss + self._dft_residual_reg_loss(model, batch)       # V58
                loss = loss + self._tier1c_dft_consistency_loss(model, batch) # V58
                loss = loss + self._kroger_distill_loss(model, batch)  # Phase 53 V53-α
                loss = loss + self._srh_coupling_loss(model, batch, pred)  # Phase 44 V8
                loss = loss + self._dcc_loss(model, batch)  # Phase 59 V59-CALM DCC

                # ── Phase 12A: latent synthesis loss ──────────────────────
                if self.synth_bank is not None:
                    sample = self.synth_bank.sample(
                        n=self.synth_per_step,
                        allowed_dois=self._cur_train_dois,
                        device=self.device,
                    )
                    if sample is not None:
                        s_emb, s_tgt, s_w = sample
                        # Apply target z-score if active (scalers fit on train fold)
                        if _tmean is not None:
                            s_tgt = (s_tgt - _tmean) / _tstd
                        s_pred = model._heads_from_fused(s_emb)
                        # Weighted MSE on synth rows (single target)
                        sq = (s_pred - s_tgt) ** 2
                        s_loss = (sq.squeeze(-1) * s_w).sum() / s_w.sum().clamp_min(1e-8)
                        loss = loss + self.synth_loss_weight * s_loss

                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                optimizer.step()

            scheduler.step()
            val_loss = self._eval_loss(
                model, val_loader, le,
                target_mean=_tmean, target_std=_tstd,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"  Fold {fold+1} early stop at epoch {epoch}.")
                    break

        # Restore best weights for this fold
        model.load_state_dict(best_model_state)

        # ── 3b. Pseudo-labeling phase 2 (optional) ────────────────────────────
        if self.pseudo_cfg.get("enabled", False) and self.mc_dropout_n > 0:
            model = self._pseudo_label_pass(
                model, train_idx, aug_dataset, le
            )

        # ── 4. MLP metrics on val ─────────────────────────────────────────────
        mlp_metrics = self._eval_metrics(
            model, val_loader,
            target_mean=fold_target_mean, target_std=fold_target_std,
        )
        self._log_fold_metrics(fold, "MLP", mlp_metrics)

        # ── 4b. Collect raw OOF predictions for post-hoc evaluation ──────────
        fold_oof = self._collect_fold_predictions(
            model, val_loader, val_idx,
            target_mean=fold_target_mean, target_std=fold_target_std,
        )

        # ── 5. XGBoost (same fold, fold-specific embeddings) ──────────────────
        xgb_metrics: dict = {}
        if self.use_xgboost:
            train_emb, train_targets_arr = self._extract_embeddings(model, train_loader)
            val_emb, val_targets_arr = self._extract_embeddings(model, val_loader)
            xgb_metrics = self._run_xgb_fold(
                train_emb, train_targets_arr, val_emb, val_targets_arr
            )
            self._log_fold_metrics(fold, "XGB", xgb_metrics)

        # ── 6. GP regression on raw XenonPy + process (no CGCNN) ─────────────
        gp_metrics: dict = {}
        if self.use_gp:
            all_targets_arr = self.dataset.get_target_array()
            all_process_arr = self.dataset.get_process_array()
            all_specs_list  = self.dataset.get_dopant_specs()
            gp_metrics = self._run_gp_fold(
                train_specs=[all_specs_list[i] for i in train_idx],
                train_process=all_process_arr[train_idx],
                train_targets=all_targets_arr[train_idx],
                val_specs=[all_specs_list[i] for i in val_idx],
                val_process=all_process_arr[val_idx],
                val_targets=all_targets_arr[val_idx],
                model=model,
            )
            self._log_fold_metrics(fold, "GP", gp_metrics)

            # 6b. Optional: GP on learned 160-dim embeddings (hybrid MLP+GP)
            if self.config.get("gp", {}).get("use_embeddings", False):
                # Reuse embeddings from XGB extraction if already computed
                if not self.use_xgboost:
                    train_emb, train_targets_arr = self._extract_embeddings(model, train_loader)
                    val_emb, val_targets_arr     = self._extract_embeddings(model, val_loader)
                gp_emb_metrics = self._run_gp_on_embeddings_fold(
                    train_emb, train_targets_arr, val_emb, val_targets_arr
                )
                gp_metrics.update(gp_emb_metrics)
                self._log_fold_metrics(fold, "GP-emb", gp_emb_metrics)

        return mlp_metrics, xgb_metrics, gp_metrics, model, best_epoch, fold_oof

    # ── OOF prediction collection ─────────────────────────────────────────────

    def _collect_fold_predictions(
        self,
        model: Ga2O3Net,
        loader: DataLoader,
        val_idx: np.ndarray,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Collect per-sample predictions for the validation fold.

        Returns a list of dicts (one per sample) with keys:
            sample_idx, dopant_label, fold (filled by run_cv),
            {col}_pred, {col}_true, {col}_std (if MC Dropout enabled)
        """
        all_preds, all_targets, all_labels, all_stds = [], [], [], []

        if self.mc_dropout_n > 0:
            for batch in loader:
                graph        = batch["graph"].to(self.device)
                process      = batch["process"].to(self.device)
                target       = batch["target"]
                dopant_specs = batch["dopant_spec"]
                mean, std = model.mc_predict(
                    graph, dopant_specs, process, self.mc_dropout_n
                )
                all_preds.append(mean.cpu().numpy())
                all_stds.append(std.cpu().numpy())
                all_targets.append(target.numpy())
                all_labels.extend(batch["dopant_label"])
        else:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    graph        = batch["graph"].to(self.device)
                    process      = batch["process"].to(self.device)
                    target       = batch["target"]
                    dopant_specs = batch["dopant_spec"]
                    emb  = model.get_embedding(graph, dopant_specs, process)
                    pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(target.numpy())
                    all_labels.extend(batch["dopant_label"])

        preds   = np.concatenate(all_preds,   axis=0)
        targets = np.concatenate(all_targets, axis=0)
        stds    = np.concatenate(all_stds, axis=0) if all_stds else None

        # Inverse-transform predictions from z-score space to original scale
        if target_mean is not None:
            preds = self._inverse_standardize(preds, target_mean, target_std)
            if stds is not None:
                # Scale uncertainty back: std_orig = std_z * target_std
                stds = stds * target_std

        rows = []
        for i, idx in enumerate(val_idx):
            row: dict = {
                "sample_idx":   int(idx),
                "dopant_label": all_labels[i],
            }
            for t, col in enumerate(self.target_cols):
                row[f"{col}_pred"] = float(preds[i, t])
                row[f"{col}_true"] = float(targets[i, t])
                if stds is not None:
                    row[f"{col}_std"] = float(stds[i, t])
            rows.append(row)
        return rows

    # ── Optimizer / reconstruction helpers ───────────────────────────────────

    def _build_optimizer(self, model: Ga2O3Net) -> Adam:
        """
        Build Adam optimizer with per-group LRs.

        When encoder_unfreeze_last_n > 0, unfrozen encoder params use
        encoder_lr (typically 20-40× smaller than base_lr) to prevent
        catastrophic forgetting.  All other params use base_lr.
        Learnable log_vars from UncertaintyWeightedLoss are included
        at base_lr.

        Phase 46 V11: when ``model.element_hypernet`` exists, its params
        get a separate group with ``hypernet_weight_decay`` (typically
        higher than the global weight_decay) to anchor it as a small
        correction rather than letting it absorb α_method's contribution.
        """
        # Phase 46 V11: identify hypernet params (if present) and route them
        # to a dedicated param group with a stronger weight_decay.
        hyper_params = []
        if getattr(model, "element_hypernet", None) is not None:
            hyper_params = [
                p for p in model.element_hypernet.parameters() if p.requires_grad
            ]
        hyper_param_ids = {id(p) for p in hyper_params}

        if self.encoder_unfreeze_last_n > 0:
            encoder_params = [
                p for p in model.encoder.parameters() if p.requires_grad
            ]
            other_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and not name.startswith("encoder.")
                and id(p) not in hyper_param_ids
            ]
            param_groups: list = [
                {"params": encoder_params, "lr": self.encoder_lr,
                 "weight_decay": self.weight_decay},
                {"params": other_params,   "lr": self.base_lr,
                 "weight_decay": self.weight_decay},
            ]
            if hyper_params:
                param_groups.append(
                    {"params": hyper_params, "lr": self.base_lr,
                     "weight_decay": self.hypernet_weight_decay}
                )
            if hasattr(self.loss_fn, "log_vars"):
                param_groups.append(
                    {"params": [self.loss_fn.log_vars], "lr": self.base_lr,
                     "weight_decay": 0.0}  # don't decay log_vars
                )
            return Adam(param_groups)
        else:
            other_trainable = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in hyper_param_ids
            ]
            if not hyper_params:
                trainable = other_trainable
                if hasattr(self.loss_fn, "log_vars"):
                    trainable = trainable + [self.loss_fn.log_vars]
                return Adam(trainable, lr=self.base_lr,
                            weight_decay=self.weight_decay)
            param_groups = [
                {"params": other_trainable, "lr": self.base_lr,
                 "weight_decay": self.weight_decay},
                {"params": hyper_params,    "lr": self.base_lr,
                 "weight_decay": self.hypernet_weight_decay},
            ]
            if hasattr(self.loss_fn, "log_vars"):
                param_groups.append(
                    {"params": [self.loss_fn.log_vars], "lr": self.base_lr,
                     "weight_decay": 0.0}
                )
            return Adam(param_groups)

    @staticmethod
    def _get_recon(
        model: Ga2O3Net,
        emb: torch.Tensor,
        dopant_specs: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute reconstruction auxiliary loss inputs.

        Returns (recon_pred, recon_target) if the model has a composition
        decoder and the composition stream has been fitted; else (None, None).
        """
        recon_pred = model.get_reconstruction(emb)
        if recon_pred is None:
            return None, None
        recon_target = model.composition_stream.get_scaled_features(dopant_specs)
        if recon_target is None:
            return None, None
        return recon_pred, recon_target

    # ── Scaler/model helpers ──────────────────────────────────────────────────

    def _make_aug_dataset(self) -> Ga2O3ExpDataset:
        aug = Ga2O3ExpDataset(
            csv_path=self.config["paths"]["experimental_csv"],
            structures_dir=self.config["paths"]["structures_dir"],
            target_cols=self.target_cols,
            augment=True,
            conc_jitter=self.config["augmentation"]["concentration_jitter"],
            label_noise_std=self.config["augmentation"]["label_noise_std"],
            method_shuffle_prob=self.config["augmentation"].get("method_shuffle_prob", 0.0),
            substrate_shuffle_prob=self.config["augmentation"].get("substrate_shuffle_prob", 0.0),
            mask_noconc_labels=self.config.get("data", {}).get("mask_noconc_labels", False),
            mask_undoped_labels=self.config.get("data", {}).get("mask_undoped_labels", False),
            min_conc_threshold=self.config.get("data", {}).get("min_conc_threshold", 0.0),
            mask_carrier_vc=self.config.get("data", {}).get("mask_carrier_vc", False),
            exclude_elements=self.config.get("data", {}).get("exclude_elements", None),
            include_elements=self.config.get("data", {}).get("include_elements", None),
            include_keep_undoped=self.config.get("data", {}).get("include_keep_undoped", True),
            process_layout=getattr(self.dataset, "process_layout", "v2"),
        )
        return aug

    def _encode_labels(self, le: LabelEncoder, labels: list[str]) -> torch.Tensor:
        """Encode string labels, mapping unseen labels to 0."""
        encoded = []
        for lbl in labels:
            if lbl in le.classes_:
                encoded.append(int(le.transform([lbl])[0]))
            else:
                encoded.append(0)
        return torch.tensor(encoded, dtype=torch.long, device=self.device)

    # ── Target standardization helpers ────────────────────────────────────────

    def _compute_target_stats(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-target mean and std from training indices (NaN-safe).

        Returns (mean, std) each of shape [num_targets].
        """
        targets = self.dataset.get_target_array()[indices]  # [N, T]
        mean = np.nanmean(targets, axis=0)
        std = np.nanstd(targets, axis=0)
        # Protect against zero-variance targets
        std = np.where(std < 1e-8, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def _standardize_target(
        self, target: torch.Tensor,
        tmean: torch.Tensor, tstd: torch.Tensor,
    ) -> torch.Tensor:
        """Z-score targets: (y - mean) / std. NaN values stay NaN."""
        return (target - tmean) / tstd

    def _inverse_standardize(
        self, arr: np.ndarray,
        mean: np.ndarray, std: np.ndarray,
    ) -> np.ndarray:
        """Inverse z-score: y * std + mean."""
        return arr * std + mean

    # ── Platt scaling (post-hoc slope correction) ─────────────────────────────

    def _fit_platt_scaling(
        self, oof_rows: list[dict]
    ) -> dict[str, tuple[float, float]]:
        """
        Fit linear calibration y_true = a * y_pred + b per target on OOF.

        Returns dict mapping target_col → (slope_a, intercept_b).
        """
        from sklearn.linear_model import LinearRegression

        platt_params: dict[str, tuple[float, float]] = {}
        for col in self.target_cols:
            preds, trues = [], []
            for r in oof_rows:
                p = r.get(f"{col}_pred", float("nan"))
                t = r.get(f"{col}_true", float("nan"))
                if not (np.isnan(p) or np.isnan(t)):
                    preds.append(p)
                    trues.append(t)
            if len(preds) < 5:
                platt_params[col] = (1.0, 0.0)
                continue
            lr = LinearRegression().fit(
                np.array(preds).reshape(-1, 1), np.array(trues)
            )
            a, b = float(lr.coef_[0]), float(lr.intercept_)
            platt_params[col] = (a, b)
            logger.info(
                f"  Platt scaling [{col}]: y_cal = {a:.3f} * y_pred + {b:.3f}"
            )
        return platt_params

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _eval_loss(
        self, model: Ga2O3Net, loader: DataLoader,
        le: LabelEncoder,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"].to(self.device)
                process = batch["process"].to(self.device)
                target = batch["target"].to(self.device)
                if target_mean is not None:
                    target = self._standardize_target(target, target_mean, target_std)
                dopant_specs = batch["dopant_spec"]
                dopant_labels = batch["dopant_label"]
                label_classes = self._encode_labels(le, dopant_labels)
                emb = model.get_embedding(graph, dopant_specs, process)
                pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                loss, _ = self.loss_fn(pred, target, emb, label_classes)
                total += loss.item()
                n += 1
        return total / max(n, 1)

    def _eval_metrics(
        self, model: Ga2O3Net, loader: DataLoader,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ) -> dict:
        all_preds, all_targets, all_stds = [], [], []
        for batch in loader:
            graph = batch["graph"].to(self.device)
            process = batch["process"].to(self.device)
            target = batch["target"]
            dopant_specs = batch["dopant_spec"]

            if self.mc_dropout_n > 0:
                # MC Dropout: ensemble mean gives better point estimate
                mean, std = model.mc_predict(
                    graph, dopant_specs, process, self.mc_dropout_n
                )
                all_preds.append(mean.cpu())
                all_stds.append(std.cpu())
            else:
                model.eval()
                with torch.no_grad():
                    emb = model.get_embedding(graph, dopant_specs, process)
                    pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                all_preds.append(pred.cpu())
            all_targets.append(target)

        preds   = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        # Inverse-transform predictions if target standardization was used
        if target_mean is not None:
            preds = self._inverse_standardize(preds, target_mean, target_std)

        metrics = {}
        for i, col in enumerate(self.target_cols):
            valid = ~np.isnan(targets[:, i])
            if valid.sum() > 0:
                for k, v in regression_metrics(
                    preds[valid, i], targets[valid, i]
                ).items():
                    metrics[f"{col}_{k}"] = v

        # Report mean uncertainty if MC Dropout was used
        if all_stds:
            stds = torch.cat(all_stds).numpy()
            for i, col in enumerate(self.target_cols):
                metrics[f"{col}_mc_std_mean"] = float(np.nanmean(stds[:, i]))

        return metrics

    def _extract_embeddings(
        self, model: Ga2O3Net, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract fused embeddings and targets for XGBoost."""
        model.eval()
        all_emb, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"].to(self.device)
                process = batch["process"].to(self.device)
                target = batch["target"]
                dopant_specs = batch["dopant_spec"]
                emb = model.get_embedding(graph, dopant_specs, process)
                all_emb.append(emb.cpu().numpy())
                all_targets.append(target.numpy())
        return (
            np.concatenate(all_emb, axis=0),
            np.concatenate(all_targets, axis=0),
        )

    # ── XGBoost (per-fold) ────────────────────────────────────────────────────

    def _run_xgb_fold(
        self,
        train_emb: np.ndarray,
        train_targets: np.ndarray,
        val_emb: np.ndarray,
        val_targets: np.ndarray,
    ) -> dict:
        """Train and evaluate XGBoost for one fold using fold-specific embeddings."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("xgboost not installed, skipping.")
            return {}

        xgb_cfg = self.config.get("xgboost", {})
        metrics = {}

        for i, col in enumerate(self.target_cols):
            y_train = train_targets[:, i]
            y_val = val_targets[:, i]

            valid_train = ~np.isnan(y_train)
            valid_val = ~np.isnan(y_val)

            if valid_train.sum() < 2 or valid_val.sum() < 1:
                logger.warning(
                    f"  XGB {col}: insufficient samples "
                    f"(train={valid_train.sum()}, val={valid_val.sum()}), skipping."
                )
                continue

            xgb = XGBRegressor(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 5),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                random_state=42,
                verbosity=0,
            )
            xgb.fit(train_emb[valid_train], y_train[valid_train])
            pred = xgb.predict(val_emb[valid_val])
            for k, v in regression_metrics(pred, y_val[valid_val]).items():
                metrics[f"{col}_{k}"] = v

        return metrics

    # ── GP regression on raw XenonPy + process features ─────────────────────

    def _run_gp_fold(
        self,
        train_specs: list[str],
        train_process: np.ndarray,
        train_targets: np.ndarray,
        val_specs: list[str],
        val_process: np.ndarray,
        val_targets: np.ndarray,
        model: "Ga2O3Net",
    ) -> dict:
        """
        Gaussian Process regression on raw XenonPy + process features.

        Does NOT use CGCNN embeddings — directly tests whether composition +
        process features (without GNN) are informative for thin-film targets.

        Pipeline per fold:
          1. Extract raw XenonPy features for each dopant spec (cached).
          2. Scale XenonPy features using the fold's composition_stream scaler.
          3. Scale continuous process dims 0-4 using process_stream scaler.
          4. Concatenate → ~295-dim; PCA to min(15, n_train//3) dims.
          5. Fit GaussianProcessRegressor (Matérn 3/2 kernel) per target.
          6. Evaluate on val fold, return regression metrics.

        Reference: Thelen et al. "Multi-Output GPR for Thin-Film Characterisation"
                   Advanced Intelligent Discovery 2026; Kim et al. npj Comp. Mat. 2020.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                Matern, ConstantKernel, WhiteKernel,
            )
            from sklearn.decomposition import PCA
            from src.models.composition_stream import spec_to_xenonpy_features
        except ImportError as exc:
            logger.warning(f"GP dependencies missing ({exc}), skipping GP fold.")
            return {}

        # ── Build feature matrices ─────────────────────────────────────────────
        def _xenonpy_matrix(specs: list[str]) -> np.ndarray:
            rows = [spec_to_xenonpy_features(s) for s in specs]
            return np.nan_to_num(np.stack(rows), nan=0.0).astype(np.float32)

        train_xeno = _xenonpy_matrix(train_specs)   # [n_train, 290]
        val_xeno   = _xenonpy_matrix(val_specs)      # [n_val,   290]

        # Scale XenonPy with the fold-specific composition scaler
        comp_scaler = getattr(model.composition_stream, "_scaler", None)
        if comp_scaler is not None:
            train_xeno = comp_scaler.transform(train_xeno).astype(np.float32)
            val_xeno   = comp_scaler.transform(val_xeno).astype(np.float32)

        # Continuous process features (dims 0..continuous_dim-1), method bits kept raw.
        # continuous_dim is 11 in Phase 2C (was 5 pre-2C); read from the stream itself.
        cont_dim = getattr(model.process_stream, "continuous_dim", 5)
        train_cont = train_process[:, :cont_dim].astype(np.float32)
        val_cont   = val_process[:,   :cont_dim].astype(np.float32)
        proc_scaler = getattr(model.process_stream, "_scaler", None)
        if proc_scaler is not None:
            train_cont = proc_scaler.transform(train_cont).astype(np.float32)
            val_cont   = proc_scaler.transform(val_cont).astype(np.float32)

        train_method = train_process[:, cont_dim:].astype(np.float32)
        val_method   = val_process[:,   cont_dim:].astype(np.float32)

        train_X = np.concatenate([train_xeno, train_cont, train_method], axis=1)
        val_X   = np.concatenate([val_xeno,   val_cont,   val_method],   axis=1)

        # ── PCA dimensionality reduction ───────────────────────────────────────
        n_valid_train = len(train_X)
        n_components = max(2, min(15, n_valid_train // 3, train_X.shape[1]))
        pca = PCA(n_components=n_components, random_state=42)
        try:
            train_X_r = pca.fit_transform(train_X)
            val_X_r   = pca.transform(val_X)
        except Exception as exc:
            logger.warning(f"GP PCA failed ({exc}), skipping GP fold.")
            return {}

        # ── Fit Ridge + GP per target ─────────────────────────────────────────
        from sklearn.linear_model import Ridge

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
        )

        metrics: dict = {}
        for i, col in enumerate(self.target_cols):
            y_train = train_targets[:, i]
            y_val   = val_targets[:,   i]
            valid_train = ~np.isnan(y_train)
            valid_val   = ~np.isnan(y_val)

            if valid_train.sum() < 4 or valid_val.sum() < 1:
                logger.warning(
                    f"  GP {col}: too few labeled samples "
                    f"(train={valid_train.sum()}, val={valid_val.sum()}), skipping."
                )
                continue

            # Ridge regression (simpler, more stable for small N)
            try:
                ridge = Ridge(alpha=1.0)
                ridge.fit(train_X_r[valid_train], y_train[valid_train])
                pred_ridge = ridge.predict(val_X_r[valid_val])
                for k, v in regression_metrics(pred_ridge, y_val[valid_val]).items():
                    metrics[f"ridge_{col}_{k}"] = v
            except Exception as exc:
                logger.warning(f"  Ridge {col} failed: {exc}")

            # Gaussian Process
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )
            try:
                gp.fit(train_X_r[valid_train], y_train[valid_train])
                pred = gp.predict(val_X_r[valid_val])
                for k, v in regression_metrics(pred, y_val[valid_val]).items():
                    metrics[f"{col}_{k}"] = v
            except Exception as exc:
                logger.warning(f"  GP {col} fit/predict failed: {exc}")

        return metrics

    # ── GP regression on MLP-extracted 160-dim embeddings (hybrid) ───────────

    def _run_gp_on_embeddings_fold(
        self,
        train_emb: np.ndarray,
        train_targets: np.ndarray,
        val_emb: np.ndarray,
        val_targets: np.ndarray,
    ) -> dict:
        """
        Hybrid MLP+GP: Gaussian Process on 160-dim learned embeddings.

        Combines learned non-linear representations from CGCNN+Composition+Process
        fusion with Bayesian regression. Uses PCA to prevent overfitting on small N.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                Matern, ConstantKernel, WhiteKernel,
            )
            from sklearn.decomposition import PCA
        except ImportError as exc:
            logger.warning(f"GP dependencies missing ({exc}), skipping GP-emb fold.")
            return {}

        n_valid_train = len(train_emb)
        n_components = max(2, min(15, n_valid_train // 3, train_emb.shape[1]))
        pca = PCA(n_components=n_components, random_state=42)
        try:
            train_X_r = pca.fit_transform(train_emb)
            val_X_r   = pca.transform(val_emb)
        except Exception as exc:
            logger.warning(f"GP-emb PCA failed ({exc}), skipping.")
            return {}

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
        )

        metrics: dict = {}
        for i, col in enumerate(self.target_cols):
            y_train = train_targets[:, i]
            y_val   = val_targets[:,   i]
            valid_train = ~np.isnan(y_train)
            valid_val   = ~np.isnan(y_val)

            if valid_train.sum() < 4 or valid_val.sum() < 1:
                continue

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )
            try:
                gp.fit(train_X_r[valid_train], y_train[valid_train])
                pred = gp.predict(val_X_r[valid_val])
                for k, v in regression_metrics(pred, y_val[valid_val]).items():
                    metrics[f"emb_{col}_{k}"] = v
            except Exception as exc:
                logger.warning(f"  GP-emb {col} fit/predict failed: {exc}")

        return metrics

    # ── Embedding-space Mixup ─────────────────────────────────────────────────

    @staticmethod
    def _mixup_batch(
        emb: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.2,
        label_kernel_bw: float = 0.0,
        primary_target_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embedding-space Mixup for regression with missing labels.

        Interpolates pairs in the fused embedding space. For each task column:
          - Both samples have a label  → interpolate label linearly.
          - Only one sample has a label → NaN in mixed target (InfoNCE only).
          - Neither has a label        → NaN preserved.

        λ ~ Beta(alpha, alpha). alpha=0.2 is standard for small datasets.
        Uses emb.detach() to avoid double-differentiation through the encoder.

        Phase 27 — C-Mixup (Yao et al. NeurIPS'22, arXiv:2210.05775):
        when ``label_kernel_bw > 0``, partner j for each anchor i is sampled
        with probability ∝ exp(-(y_i - y_j)² / (2·bw²)) on the primary target,
        i.e. label-proximity-weighted instead of uniform. This fixes the
        regression-mixup pathology where linear label interpolation across
        distant labels is incorrect.
        """
        B = emb.size(0)
        if B < 2:
            return emb, target

        lam = float(np.random.beta(alpha, alpha))

        if label_kernel_bw and label_kernel_bw > 0.0:
            # C-Mixup: label-aware partner sampling
            y = target[:, primary_target_idx]
            valid = ~torch.isnan(y)
            if valid.sum() >= 2:
                y_safe = torch.where(valid, y, torch.zeros_like(y))
                # pairwise label distance and Gaussian similarity
                diff = y_safe.unsqueeze(0) - y_safe.unsqueeze(1)        # [B, B]
                sims = torch.exp(-(diff ** 2) / (2.0 * label_kernel_bw ** 2))
                # mask self-pairs and invalid endpoints
                eye = torch.eye(B, dtype=torch.bool, device=emb.device)
                pair_mask = (~eye) & valid.unsqueeze(0) & valid.unsqueeze(1)
                sims = sims * pair_mask.float()
                row_sum = sims.sum(dim=1, keepdim=True)
                # rows with no valid partner: fall back to uniform-random
                no_partner = (row_sum.squeeze(-1) <= 1e-8)
                row_sum = row_sum.clamp(min=1e-8)
                probs = sims / row_sum
                perm = torch.multinomial(probs.clamp(min=1e-12), 1).squeeze(-1)
                if no_partner.any():
                    fallback = torch.randperm(B, device=emb.device)
                    perm[no_partner] = fallback[no_partner]
            else:
                perm = torch.randperm(B, device=emb.device)
        else:
            perm = torch.randperm(B, device=emb.device)

        mixed_emb = lam * emb + (1.0 - lam) * emb[perm]

        mixed_target = target.clone()
        t2 = target[perm]
        for t in range(target.size(1)):
            both = (~torch.isnan(target[:, t])) & (~torch.isnan(t2[:, t]))
            mixed_target[both, t] = (
                lam * target[both, t] + (1.0 - lam) * t2[both, t]
            )
            only_one = (
                (~torch.isnan(target[:, t])) ^ (~torch.isnan(t2[:, t]))
            )
            mixed_target[only_one, t] = float("nan")

        return mixed_emb, mixed_target

    # ── Pseudo-labeling phase 2 ───────────────────────────────────────────────

    def _pseudo_label_pass(
        self,
        model: Ga2O3Net,
        train_idx: np.ndarray,
        aug_dataset: Ga2O3ExpDataset,
        le: LabelEncoder,
    ) -> Ga2O3Net:
        """
        Phase-2 pseudo-labeling (Cascante-Bonilla et al., AAAI 2021 style).

        After phase-1 training:
          1. Run MC Dropout on all training-fold samples → mean + std per target.
          2. For each task: accept pseudo-labels where std < std_threshold.
          3. Build PseudoLabeledSubset with real labels + accepted pseudo-labels.
          4. Train for phase2_epochs at reduced LR on the expanded set.

        Only training-fold samples are ever pseudo-labeled; val/test indices are
        never touched.
        """
        std_threshold = self.pseudo_cfg.get("std_threshold", 0.3)
        phase2_epochs = self.pseudo_cfg.get("phase2_epochs", 500)
        lr_scale      = self.pseudo_cfg.get("phase2_lr_scale", 0.1)
        mc_passes     = self.pseudo_cfg.get("mc_passes", self.mc_dropout_n)

        # Get ground-truth targets for training fold (NaN where missing)
        all_targets = self.dataset.get_target_array()
        train_targets = all_targets[train_idx]  # [n_train, T]
        pseudo_targets = train_targets.copy()

        # Build a loader for the training fold (no augmentation for inference)
        infer_loader = DataLoader(
            Subset(self.dataset, train_idx.tolist()),
            batch_size=max(self.batch_size, 4),
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Collect MC Dropout predictions over the training fold
        all_means, all_stds = [], []
        for batch in infer_loader:
            graph        = batch["graph"].to(self.device)
            process      = batch["process"].to(self.device)
            dopant_specs = batch["dopant_spec"]
            mean, std = model.mc_predict(graph, dopant_specs, process, mc_passes)
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())

        means = np.concatenate(all_means, axis=0)   # [n_train, T]
        stds  = np.concatenate(all_stds,  axis=0)   # [n_train, T]

        # Accept pseudo-labels
        total_new = 0
        for t, col in enumerate(self.target_cols):
            missing          = np.isnan(train_targets[:, t])
            low_uncertainty  = stds[:, t] < std_threshold
            accept           = missing & low_uncertainty
            pseudo_targets[accept, t] = means[accept, t]
            n_accepted = int(accept.sum())
            total_new += n_accepted
            logger.info(
                f"  Pseudo-label [{col}]: {n_accepted}/{missing.sum()} accepted "
                f"(std_threshold={std_threshold:.2f}, "
                f"std_range=[{stds[missing, t].min():.3f}, {stds[missing, t].max():.3f}])"
            )

        if total_new < 1:
            logger.info("  No pseudo-labels accepted; skipping phase 2.")
            return model

        # Build PseudoLabeledSubset and phase-2 loader
        pl_subset = PseudoLabeledSubset(aug_dataset, train_idx.tolist(), pseudo_targets)
        pl_loader = DataLoader(
            pl_subset,
            batch_size=max(self.batch_size, 4),
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Phase-2 training at reduced LR (continue from phase-1 weights)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if hasattr(self.loss_fn, "log_vars"):
            trainable = trainable + [self.loss_fn.log_vars]
        p2_optimizer = Adam(trainable, lr=self.base_lr * lr_scale)

        logger.info(
            f"  Phase-2 training for {phase2_epochs} epochs "
            f"(lr={self.base_lr * lr_scale:.2e}, {len(pl_subset)} samples)"
        )
        for _ in range(phase2_epochs):
            model.train()
            for batch in pl_loader:
                graph        = batch["graph"].to(self.device)
                process      = batch["process"].to(self.device)
                target       = batch["target"].to(self.device)
                dopant_specs = batch["dopant_spec"]
                dopant_labels = batch["dopant_label"]
                label_classes = self._encode_labels(le, dopant_labels)

                p2_optimizer.zero_grad()
                emb = model.get_embedding(graph, dopant_specs, process)

                sample_w = batch.get("sample_weight")
                if sample_w is not None:
                    sample_w = sample_w.to(self.device)

                if self.mixup_alpha > 0.0:
                    emb_in, target_in = self._mixup_batch(
                        emb.detach(), target, self.mixup_alpha,
                        label_kernel_bw=self.mixup_label_kernel_bw,
                        primary_target_idx=self.mixup_primary_target_idx,
                    )
                    pred = model._heads_from_fused(emb_in, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target_in, emb_in, label_classes, model
                    )
                else:
                    pred = model._heads_from_fused(emb, dopant_specs=dopant_specs, process=process)
                    loss, _ = self.loss_fn(
                        pred, target, emb, label_classes, model,
                        sample_weights=sample_w,
                    )

                loss = loss + self._family_cos_loss(model)   # Phase 6A
                loss = loss + self._monotonic_constraint_loss(model, batch)   # Phase 6C
                loss = loss + self._monotonic_vc_concentration_loss(model, batch)  # Phase 17
                loss = loss + self._monotonic_vc_within_doi_loss(model, batch)  # Phase 20
                loss = loss + self._atmosphere_ordinal_vc_loss(model, batch)  # Phase 32
                loss = loss + self._atmosphere_ordinal_pdr_loss(model, batch)  # Phase 28-PDR
                loss = loss + self._anneal_temperature_sign_loss(model, batch)  # Phase 36
                loss = loss + self._element_class_contrastive_loss(model, batch)  # Phase 35
                loss = loss + self._cross_element_ranking_loss(model, batch)  # Phase 23
                loss = loss + self._cross_method_dopant_pull_loss(model, batch)  # Phase 42
                loss = loss + self._encoder_distill_loss(model, batch)  # Phase 43
                loss = loss + self._contrastive_pdr_vc_loss(model, batch)  # Phase 53 V53-ζ
                loss = loss + self._pdr_anchor_sincere_loss(model, batch)   # Phase 54 V54-A1
                loss = loss + self._pdr_self_distill_loss(model, batch)     # Phase 54 V54-A1
                loss = loss + self._process_axis_infonce_loss(model, batch) # Phase 54 V54-A2
                loss = loss + self._pysr_dopant_soft_loss(model)            # Phase 54 V54-C1
                loss = loss + self._charge_vo_distill_loss(model, batch)    # Phase 54 V54-B1
                loss = loss + self._vga_distill_loss(model, batch)          # Phase 54 V54-B1
                loss = loss + self._dft_scan_distill_loss(model, batch)      # V57-B1-Quartet+1
                loss = loss + self._mace_latent_distill_loss(model, batch)   # V57-MACE-Latent
                loss = loss + self._dft_residual_reg_loss(model, batch)       # V58
                loss = loss + self._tier1c_dft_consistency_loss(model, batch) # V58
                loss = loss + self._kroger_distill_loss(model, batch)  # Phase 53 V53-α
                loss = loss + self._srh_coupling_loss(model, batch, pred)  # Phase 44 V8
                loss = loss + self._dcc_loss(model, batch)  # Phase 59 V59-CALM DCC
                loss = loss + self._physics_envelope_loss_pdr(model, batch, pred)  # Phase 25-soft
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                p2_optimizer.step()

        return model

    # ── Aggregate / logging ───────────────────────────────────────────────────

    @staticmethod
    def _aggregate(fold_metrics: dict[str, list], prefix: str) -> dict:
        summary = {}
        for k, vals in fold_metrics.items():
            summary[f"{prefix}_{k}_mean"] = float(np.mean(vals))
            summary[f"{prefix}_{k}_std"] = float(np.std(vals))
        return summary

    @staticmethod
    def _log_fold_metrics(fold: int, head: str, metrics: dict):
        if not metrics:
            return
        logger.info(f"  Fold {fold+1} [{head}] metrics:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.4f}")
