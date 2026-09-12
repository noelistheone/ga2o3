"""
Loss functions for Ga2O3-Net training.

Available losses:
  MSEWithL2Loss            — standard regression + weight decay
  SupervisedInfoNCELoss    — push different dopant embeddings apart
  CombinedLoss             — MSE + InfoNCE + L2 (Stage 3 default)
  UncertaintyWeightedLoss  — per-task homoscedastic uncertainty weighting
                             (Kendall & Gal 2018) + InfoNCE + L2
                             + optional composition reconstruction auxiliary loss
                             (semi-supervised: uses ALL samples regardless of labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWithL2Loss(nn.Module):
    """
    MSE regression loss + explicit L2 weight regularization.

    Note: l2_lambda doubles up Adam's weight_decay; use one or the other.
    Setting l2_lambda=0 reduces to plain MSE.
    """

    def __init__(self, l2_lambda: float = 1e-4):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict]:
        mse = self.mse(pred, target)
        l2 = torch.tensor(0.0, device=pred.device)
        if model is not None and self.l2_lambda > 0:
            for p in model.parameters():
                if p.requires_grad:
                    l2 = l2 + p.norm(2)
        total = mse + self.l2_lambda * l2
        return total, {"mse": mse.item(), "l2": l2.item()}


class SupervisedInfoNCELoss(nn.Module):
    """
    Supervised contrastive (InfoNCE) loss that pulls together embeddings
    from the same dopant class and pushes apart different classes.

    Uses cosine similarity with a temperature parameter.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.3):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embedding vectors.
            labels: [B] integer class labels (e.g., dopant element index).

        Returns:
            Scalar loss.
        """
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # L2-normalize
        emb = F.normalize(embeddings, dim=1)

        # Pairwise cosine similarity [B, B]
        sim = torch.matmul(emb, emb.T) / self.temperature

        # Masks
        labels = labels.view(-1)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
        mask_pos.fill_diagonal_(0.0)   # exclude self
        mask_neg = 1.0 - (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # noqa: F841

        # Remove self from denominator
        eye = torch.eye(B, device=embeddings.device)
        exp_sim = torch.exp(sim) * (1 - eye)

        pos_sum = (exp_sim * mask_pos).sum(dim=1)        # [B]
        total_sum = exp_sim.sum(dim=1) + 1e-8            # [B]

        # For samples with no positive pair, loss = 0
        has_positive = mask_pos.sum(dim=1) > 0           # [B]
        log_prob = torch.log(pos_sum / total_sum + 1e-8) * has_positive.float()
        loss = -log_prob.mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Stage 3 combined loss: MSE + InfoNCE + L2 + optional reconstruction
                           + optional MoE load-balancing.

    Args:
        mse_weight:    Weight for MSE regression loss.
        infonce_weight: Weight for supervised contrastive loss.
        l2_lambda:     Explicit L2 weight regularization coefficient.
        temperature:   InfoNCE temperature.
        recon_weight:  Weight for composition reconstruction auxiliary loss.
                       Set to 0.0 to disable. Uses all samples (semi-supervised).
        moe_lb_weight: Weight for MoE load-balancing entropy loss
                       (max_entropy − actual_entropy of avg gate usage).
                       Set to 0.0 to disable (effective when not using MoE heads).
    """

    def __init__(
        self,
        mse_weight: float = 0.85,
        infonce_weight: float = 0.15,
        l2_lambda: float = 1e-3,
        temperature: float = 0.3,
        recon_weight: float = 0.0,
        moe_lb_weight: float = 0.0,
        regression_loss: str = "mse",
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.infonce_weight = infonce_weight
        self.l2_lambda = l2_lambda
        self.recon_weight = recon_weight
        self.moe_lb_weight = moe_lb_weight
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta
        # 12K: Huber loss (robust to outlier targets / synth noise)
        if regression_loss == "huber":
            self.mse = nn.HuberLoss(delta=huber_delta)
        else:
            self.mse = nn.MSELoss()
        self.infonce = SupervisedInfoNCELoss(temperature)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        embeddings: torch.Tensor,
        label_classes: torch.Tensor,
        model: nn.Module | None = None,
        recon_pred: torch.Tensor | None = None,
        recon_target: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:         [B, num_targets] model predictions.
            target:       [B, num_targets] ground-truth labels (NaN = missing).
            embeddings:   [B, D] fused embeddings for InfoNCE.
            label_classes:[B] integer class index (dopant element index).
            model:        optional model reference for explicit L2.
            recon_pred:   [B, raw_dim] decoder output (optional).
            recon_target: [B, raw_dim] scaled XenonPy features (optional).
            sample_weights: [B] optional per-sample MSE weight. Default 1.0 each.

        Returns:
            (total_loss, metrics_dict)
        """
        valid = ~torch.isnan(target)
        mse_loss = torch.tensor(0.0, device=pred.device)
        if valid.any():
            if sample_weights is not None:
                # Mask first so NaN positions never enter autograd.
                w = sample_weights.to(pred.device).unsqueeze(1).expand_as(target)
                p_v = pred[valid]
                t_v = target[valid]
                w_v = w[valid]
                # 12K: Huber loss is robust to outlier targets / noisy synth.
                if self.regression_loss == "huber":
                    res = p_v - t_v
                    abs_res = res.abs()
                    delta = self.huber_delta
                    quad = torch.minimum(abs_res, torch.tensor(delta, device=res.device))
                    lin = abs_res - quad
                    sq_v = 0.5 * quad ** 2 + delta * lin
                else:
                    sq_v = (p_v - t_v) ** 2
                mse_loss = (sq_v * w_v).sum() / w_v.sum().clamp_min(1e-8)
            else:
                mse_loss = self.mse(pred[valid], target[valid])

        infonce_loss = self.infonce(embeddings, label_classes)

        l2 = torch.tensor(0.0, device=pred.device)
        if model is not None and self.l2_lambda > 0:
            for p in model.parameters():
                if p.requires_grad:
                    l2 = l2 + p.norm(2)

        total = (
            self.mse_weight * mse_loss
            + self.infonce_weight * infonce_loss
            + self.l2_lambda * l2
        )

        metrics = {
            "total": total.item(),
            "mse": mse_loss.item(),
            "infonce": infonce_loss.item(),
            "l2": l2.item(),
        }

        # Composition reconstruction auxiliary loss (semi-supervised)
        if (self.recon_weight > 0.0
                and recon_pred is not None
                and recon_target is not None):
            recon_loss = F.mse_loss(recon_pred, recon_target)
            total = total + self.recon_weight * recon_loss
            metrics["recon"] = recon_loss.item()

        # MoE load-balancing loss (only effective when model has MoE heads)
        if self.moe_lb_weight > 0.0 and model is not None:
            from src.models.moe_head import MoEHead, load_balancing_loss
            lb_total = torch.zeros((), device=pred.device)
            n_lb = 0
            for head in getattr(model, "heads", {}).values() if hasattr(model, "heads") else []:
                if isinstance(head, MoEHead):
                    gw = head.get_gate_weights()
                    if gw is not None:
                        lb_total = lb_total + load_balancing_loss(gw)
                        n_lb += 1
            if n_lb > 0:
                lb_term = lb_total / n_lb
                total = total + self.moe_lb_weight * lb_term
                metrics["moe_lb"] = lb_term.item()

        return total, metrics


class UncertaintyWeightedLoss(nn.Module):
    """
    Per-task homoscedastic uncertainty weighting (Kendall & Gal, NeurIPS 2018).

    Learns one log-variance parameter per task.  For task t:
        L_t = MSE_t / (2·exp(log_var_t)) + 0.5·log_var_t

    High model uncertainty for a task → log_var_t rises → MSE contribution
    is down-weighted automatically.  This self-balances tasks with very
    different amounts of labeled data (e.g. 54 vs 14 samples).

    NaN masking is applied per task (a sample may be labeled for one target
    but not the other).

    Also supports an optional composition reconstruction auxiliary loss that
    provides gradient signal from ALL samples, including those without
    regression labels (semi-supervised; ASGN KDD 2020 style).

    Args:
        num_tasks:      Number of regression targets.
        infonce_weight: Weight for InfoNCE contrastive loss.
        l2_lambda:      Explicit L2 weight regularisation coefficient.
        temperature:    InfoNCE temperature.
        init_log_var:   Initial value of log σ² for each task (default 0 → σ²=1).
        recon_weight:   Weight for composition reconstruction auxiliary loss.
                        Set to 0.0 to disable.
    """

    def __init__(
        self,
        num_tasks: int = 2,
        infonce_weight: float = 0.15,
        l2_lambda: float = 1e-3,
        temperature: float = 0.3,
        init_log_var: float = 0.0,
        recon_weight: float = 0.0,
    ):
        super().__init__()
        # Learnable per-task log variance; shape [num_tasks]
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))
        self.infonce_weight = infonce_weight
        self.l2_lambda = l2_lambda
        self.recon_weight = recon_weight
        self.infonce = SupervisedInfoNCELoss(temperature)
        self.num_tasks = num_tasks

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        embeddings: torch.Tensor,
        label_classes: torch.Tensor,
        model: nn.Module | None = None,
        recon_pred: torch.Tensor | None = None,
        recon_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:         [B, num_tasks] predictions.
            target:       [B, num_tasks] ground-truth labels (NaN = missing).
            embeddings:   [B, D] fused embeddings for InfoNCE.
            label_classes:[B] integer class index.
            model:        optional for L2 penalty.
            recon_pred:   [B, raw_dim] decoder output (optional).
            recon_target: [B, raw_dim] scaled XenonPy features (optional).

        Returns:
            (total_loss, metrics_dict)
        """
        task_losses = []
        for t in range(self.num_tasks):
            valid = ~torch.isnan(target[:, t])
            if valid.any():
                mse_t = F.mse_loss(pred[valid, t], target[valid, t])
                # Precision-weighted MSE + log-normalisation term
                task_losses.append(
                    0.5 * torch.exp(-self.log_vars[t]) * mse_t
                    + 0.5 * self.log_vars[t]
                )
            else:
                task_losses.append(torch.tensor(0.0, device=pred.device))

        mse_weighted = sum(task_losses)

        infonce_loss = self.infonce(embeddings, label_classes)

        l2 = torch.tensor(0.0, device=pred.device)
        if model is not None and self.l2_lambda > 0:
            for p in model.parameters():
                if p.requires_grad:
                    l2 = l2 + p.norm(2)

        total = mse_weighted + self.infonce_weight * infonce_loss + self.l2_lambda * l2

        metrics = {
            "total": total.item(),
            "mse_weighted": mse_weighted.item(),
            "infonce": infonce_loss.item(),
            "l2": l2.item(),
        }
        for t in range(self.num_tasks):
            metrics[f"log_var_{t}"] = self.log_vars[t].item()

        # Composition reconstruction auxiliary loss (semi-supervised)
        if (self.recon_weight > 0.0
                and recon_pred is not None
                and recon_target is not None):
            recon_loss = F.mse_loss(recon_pred, recon_target)
            total = total + self.recon_weight * recon_loss
            metrics["recon"] = recon_loss.item()

        return total, metrics
