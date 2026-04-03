import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class FairSupConLoss(nn.Module):
    """Fair SupCon loss (Khosla et al., NeurIPS 2020). See README for derivation.

    Positives: same label + different sensitive attribute (P_Fair).
    Denominator: P_Fair ∪ negatives; same-label-same-sensitive pairs are excluded.
    Falls back to standard SupCon when sensitives=None.
    """

    def __init__(self, temperature=cfg.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, sensitives=None):
        """
        Args:
            features:   [B, feat_dim], L2-normalized embeddings.
            labels:     [B], target labels.
            sensitives: [B] or None, sensitive attribute.
        Returns:
            scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        # mask[i,j]=1 iff y_i == y_j
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = 1.0 - torch.eye(batch_size, device=device)

        if sensitives is not None:
            sensitives = sensitives.contiguous().view(-1, 1)
            diff_sensitive = torch.ne(sensitives, sensitives.T).float().to(device)
            # P_Fair(i): same label, different sensitive, not self
            mask_pos = mask * diff_sensitive * logits_mask
            # D(i) = P_Fair(i) ∪ N(i); exclude same-label-same-sensitive from denom
            same_label_same_sens = mask * (1.0 - diff_sensitive) * logits_mask
            denom_mask = logits_mask - same_label_same_sens
        else:
            # fallback to standard SupCon
            mask_pos = mask * logits_mask
            denom_mask = logits_mask

        # sim(i,k)/tau with numerical stability (subtract row max)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # log-softmax over D(i) only
        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # average log-prob over fair positives, skip anchors with no positives
        mask_pos_pairs = mask_pos.sum(1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos_pairs + 1e-8)

        loss = -mean_log_prob_pos[mask_pos_pairs > 0].mean()
        return loss


class GroupWeightedCrossEntropyLoss(nn.Module):
    """Group-weighted CE loss for reweighting mode."""

    def __init__(self, group_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("group_weights", group_weights.float())

    def forward(self, logits, targets, sensitives):
        groups = targets * 2 + sensitives
        weights = self.group_weights[groups]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        return (ce_loss * weights).mean()


class TotalLoss(nn.Module):
    """L_total = L_CE + lambda * L_FSC."""

    def __init__(self, lambda_con=cfg.LAMBDA_CON, temperature=cfg.TEMPERATURE, group_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss() if group_weights is None else GroupWeightedCrossEntropyLoss(group_weights)
        self.supcon = FairSupConLoss(temperature=temperature)
        self.lam = lambda_con
        self.use_group_reweight = group_weights is not None

    def forward(self, logits, embeddings, targets, sensitives=None):
        if self.use_group_reweight:
            if sensitives is None:
                raise ValueError("sensitives is required when group reweighting is enabled")
            ce = self.ce(logits, targets, sensitives)
        else:
            ce = self.ce(logits, targets)
        con = self.supcon(embeddings, targets, sensitives)
        return ce + self.lam * con, ce, con
