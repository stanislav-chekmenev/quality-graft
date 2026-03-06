"""Utilities for pLDDT binning, matching Boltz1 training loss conventions."""

import torch
from torch import Tensor

NUM_PLDDT_BINS = 50


def plddt_to_bin(plddt: Tensor, num_bins: int = NUM_PLDDT_BINS) -> Tensor:
    """Convert continuous pLDDT values (0-1 scale) to bin indices.

    Uses the same binning as Boltz training loss:
        bin_index = floor(plddt * num_bins), clamped to [0, num_bins-1]

    Args:
        plddt: Tensor of pLDDT values in [0, 1]. Any shape.
        num_bins: Number of bins (default 50).

    Returns:
        LongTensor of bin indices, same shape as input.
    """
    bin_index = torch.floor(plddt * num_bins).long()
    return torch.clamp(bin_index, min=0, max=num_bins - 1)


def bin_to_plddt(bin_index: Tensor, num_bins: int = NUM_PLDDT_BINS) -> Tensor:
    """Convert bin indices back to bin center values.

    Args:
        bin_index: Tensor of bin indices in [0, num_bins-1]. Any shape.
        num_bins: Number of bins (default 50).

    Returns:
        Float tensor of pLDDT values at bin centers, same shape as input.
    """
    bin_width = 1.0 / num_bins
    return (bin_index.float() + 0.5) * bin_width
