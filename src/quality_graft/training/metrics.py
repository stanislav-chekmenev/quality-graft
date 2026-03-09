"""Validation metrics for pLDDT prediction.

All metrics are computed per-protein (masked), then averaged across the batch.
"""

import torch
from torch import Tensor


def _bin_centers(num_bins: int = 50) -> Tensor:
    """Return bin center values in [0, 1] for the given number of bins."""
    bin_width = 1.0 / num_bins
    return torch.arange(num_bins, dtype=torch.float32) * bin_width + bin_width / 2


def _logits_to_continuous(logits: Tensor, num_bins: int = 50) -> Tensor:
    """Convert bin logits to continuous pLDDT via expected value.

    Args:
        logits: [b, n, num_bins]

    Returns:
        [b, n] continuous pLDDT in [0, 1]
    """
    probs = torch.softmax(logits, dim=-1)  # [b, n, num_bins]
    centers = _bin_centers(num_bins).to(probs.device, probs.dtype)  # [num_bins]
    return (probs * centers).sum(dim=-1)  # [b, n]


def _labels_to_continuous(labels: Tensor, num_bins: int = 50) -> Tensor:
    """Convert bin indices to continuous pLDDT via bin centers.

    Args:
        labels: [b, n] long tensor of bin indices

    Returns:
        [b, n] continuous pLDDT in [0, 1]
    """
    centers = _bin_centers(num_bins).to(labels.device)
    return centers[labels]


def plddt_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Masked top-1 bin prediction accuracy.

    Args:
        logits: [b, n, 50] predicted logits
        labels: [b, n] ground truth bin indices
        mask: [b, n] float mask (1 = valid, 0 = padding)

    Returns:
        Scalar accuracy averaged over all valid residues.
    """
    preds = logits.argmax(dim=-1)  # [b, n]
    correct = (preds == labels).float() * mask
    return correct.sum() / mask.sum().clamp(min=1)


def plddt_mae(logits: Tensor, labels: Tensor, mask: Tensor, num_bins: int = 50) -> Tensor:
    """Masked mean absolute error between predicted and ground truth pLDDT.

    Converts both to continuous [0, 1] via bin centers, then computes MAE.

    Args:
        logits: [b, n, num_bins] predicted logits
        labels: [b, n] ground truth bin indices
        mask: [b, n] float mask
        num_bins: number of pLDDT bins

    Returns:
        Scalar MAE averaged over all valid residues.
    """
    pred_continuous = _logits_to_continuous(logits, num_bins)
    target_continuous = _labels_to_continuous(labels, num_bins)
    ae = (pred_continuous - target_continuous).abs() * mask
    return ae.sum() / mask.sum().clamp(min=1)


def pearson_r(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Pearson correlation, averaged across the batch.

    Args:
        pred: [b, n] predicted continuous pLDDT
        target: [b, n] ground truth continuous pLDDT
        mask: [b, n] float mask

    Returns:
        Scalar mean Pearson r across proteins.
    """
    batch_size = pred.shape[0]
    rs = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred[i][m]
        t = target[i][m]
        if p.numel() < 3:
            continue
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        num = (p_centered * t_centered).sum()
        den = (p_centered.pow(2).sum() * t_centered.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(rs).mean()


def _rank(x: Tensor) -> Tensor:
    """Compute ranks (1-based) for a 1-D tensor. Ties get averaged rank."""
    sorted_indices = x.argsort()
    ranks = torch.empty_like(x)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    return ranks


def spearman_r(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Spearman rank correlation, averaged across the batch.

    Args:
        pred: [b, n] predicted continuous pLDDT
        target: [b, n] ground truth continuous pLDDT
        mask: [b, n] float mask

    Returns:
        Scalar mean Spearman r across proteins.
    """
    batch_size = pred.shape[0]
    rs = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred[i][m]
        t = target[i][m]
        if p.numel() < 3:
            continue
        p_ranked = _rank(p)
        t_ranked = _rank(t)
        p_centered = p_ranked - p_ranked.mean()
        t_centered = t_ranked - t_ranked.mean()
        num = (p_centered * t_centered).sum()
        den = (p_centered.pow(2).sum() * t_centered.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(rs).mean()