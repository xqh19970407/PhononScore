"""Losses and score utilities for phonon-stability ranking experiments."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def make_threshold_labels(target: Tensor, thresholds: Sequence[float]) -> Tensor:
    """Return binary labels for target > threshold for every threshold."""
    values = target.detach().view(-1, 1)
    thresh = torch.as_tensor(thresholds, dtype=values.dtype, device=values.device).view(1, -1)
    return (values > thresh).to(dtype=values.dtype)


def classification_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    mode: str = "bce",
    focal_gamma: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Multi-threshold classification loss with optional class balance/focal weighting."""
    mode = mode.lower()
    if mode not in {"bce", "balanced_bce", "focal", "balanced_focal"}:
        raise ValueError(f"unknown classification loss mode: {mode}")

    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)

    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if "balanced" in mode:
        positives = labels.sum(dim=0)
        negatives = labels.shape[0] - positives
        pos_weight = (negatives + eps) / (positives + eps)
        element_weight = torch.where(labels > 0.5, pos_weight.view(1, -1), torch.ones_like(labels))
        loss = loss * element_weight
    if "focal" in mode:
        prob = torch.sigmoid(logits)
        p_t = torch.where(labels > 0.5, prob, 1.0 - prob)
        loss = loss * (1.0 - p_t).clamp_min(0.0).pow(float(focal_gamma))
    return loss.mean()


def ordinal_monotonic_loss(logits: Tensor, margin: float = 0.0) -> Tensor:
    """Penalize strict-threshold logits that exceed looser-threshold logits."""
    if logits.shape[-1] < 2:
        return logits.sum() * 0.0
    violation = logits[..., :-1] - logits[..., 1:] + float(margin)
    return F.relu(violation).mean()


def mdn_negative_log_likelihood(pi: Tensor, sigma: Tensor, mu: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    """Mixture-density negative log likelihood for scalar pair distances."""
    y = target
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    log_norm = -torch.log(sigma.clamp_min(eps)) - 0.5 * torch.log(
        torch.as_tensor(2.0 * torch.pi, dtype=sigma.dtype, device=sigma.device)
    )
    log_prob = log_norm - 0.5 * ((y.expand_as(mu) - mu) / sigma.clamp_min(eps)).pow(2)
    return -torch.logsumexp(torch.log(pi.clamp_min(eps)) + log_prob, dim=-1)


def mdn_log_probability(pi: Tensor, sigma: Tensor, mu: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    """Mixture-density log probability for scalar pair distances."""
    return -mdn_negative_log_likelihood(pi=pi, sigma=sigma, mu=mu, target=target, eps=eps)


def pairwise_ranking_loss(score: Tensor, target: Tensor, min_label_gap: float = 0.05, max_pairs: int | None = None) -> Tensor:
    """RankNet-style loss that prefers higher scores for higher targets."""
    score = score.view(-1)
    target = target.view(-1)
    if score.numel() < 2:
        return score.sum() * 0.0

    diff_y = target.unsqueeze(1) - target.unsqueeze(0)
    mask = diff_y > float(min_label_gap)
    if not torch.any(mask):
        return score.sum() * 0.0

    diff_score = score.unsqueeze(1) - score.unsqueeze(0)
    margins = diff_score[mask]
    if max_pairs is not None and margins.numel() > max_pairs:
        index = torch.randperm(margins.numel(), device=margins.device)[:max_pairs]
        margins = margins[index]
    return F.softplus(-margins).mean()


def threshold_aware_ranking_loss(
    score: Tensor,
    target: Tensor,
    thresholds: Sequence[float],
    *,
    max_pairs: int | None = None,
) -> Tensor:
    """Ranking loss focused on sample pairs split by at least one stability threshold."""
    score = score.view(-1)
    target = target.view(-1)
    if score.numel() < 2:
        return score.sum() * 0.0

    labels = make_threshold_labels(target, thresholds)
    diff_label = labels.unsqueeze(1) - labels.unsqueeze(0)
    pair_weight = diff_label.clamp_min(0).sum(dim=-1)
    mask = pair_weight > 0
    if not torch.any(mask):
        return score.sum() * 0.0

    diff_score = score.unsqueeze(1) - score.unsqueeze(0)
    margins = diff_score[mask]
    weights = pair_weight[mask]
    if max_pairs is not None and margins.numel() > max_pairs:
        index = torch.randperm(margins.numel(), device=margins.device)[:max_pairs]
        margins = margins[index]
        weights = weights[index]
    loss = F.softplus(-margins) * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def combine_final_score(
    min_freq_pred: Tensor,
    *,
    pair_geometry_score: Tensor | None = None,
    multi_threshold_logits: Tensor | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Tensor:
    """Combine regression, periodic pair geometry, and threshold heads into one score."""
    score = min_freq_pred.view(-1)
    if pair_geometry_score is not None and alpha != 0:
        score = score + float(alpha) * pair_geometry_score.view(-1)
    if multi_threshold_logits is not None and beta != 0:
        threshold_score = torch.sigmoid(multi_threshold_logits).mean(dim=-1)
        score = score + float(beta) * threshold_score.view(-1)
    return score
