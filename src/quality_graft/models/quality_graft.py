"""Full Quality-Graft assembled model.

Wires together the three core components:

    La-Proteina (frozen) → Adaptor (trainable) → Confidence Head → pLDDT

The confidence head can be either:
- **BoltzConfidenceHead** (frozen, 152.7M params): original 48-block pairformer
- **StudentConfidenceHead** (trainable, ~11-14M params): distilled student network

The model accepts a protein-structure batch (coordinates, masks, residue types)
and produces per-residue quality predictions (pLDDT, PDE, resolved logits).

Architecture reference: plans/architecture.md Section 5.4
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from quality_graft.models.adaptor import AdaptorModule


class QualityGraft(nn.Module):
    """Full Quality-Graft model.

    La-Proteina (frozen) → Adaptor (trainable) → Confidence Head → pLDDT

    The confidence head can be either a frozen ``BoltzConfidenceHead``
    (original architecture) or a trainable ``StudentConfidenceHead``
    (distillation architecture).  Both share the same forward interface:
    ``(s, z, mask) -> dict[str, Tensor]``.

    Parameters
    ----------
    la_proteina : nn.Module
        Frozen La-Proteina wrapper that extracts intermediate representations.
    adaptor : AdaptorModule
        Trainable adaptor that projects La-Proteina representations into
        Boltz1 dimension space.
    confidence_head : nn.Module
        Confidence head (BoltzConfidenceHead or StudentConfidenceHead).
    """

    def __init__(
        self,
        la_proteina: nn.Module,
        adaptor: AdaptorModule,
        confidence_head: nn.Module,
    ) -> None:
        super().__init__()
        self.la_proteina = la_proteina
        self.adaptor = adaptor
        self.confidence_head = confidence_head

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """End-to-end forward: raw batch → quality predictions.

        Parameters
        ----------
        batch : dict
            Protein-structure batch accepted by
            :meth:`LaProteinaWrapper.forward`.  Must contain at minimum
            ``coords_nm``, ``coord_mask`` (or ``mask``), and
            ``residue_type``.

        Returns
        -------
        dict[str, Tensor]
            ``plddt_logits``    — ``[b, n, 50]``   per-residue pLDDT bin logits.
            ``pde_logits``      — ``[b, n, n, 64]`` pairwise distance error logits.
            ``resolved_logits`` — ``[b, n, 2]``    per-residue resolved logits.
        """
        # 1. Extract La-Proteina representations (frozen, no grad)
        reprs = self.la_proteina(batch)

        # 2. Recover the mask produced by the La-Proteina wrapper's flow
        #    matcher processing.  Convert to float for adaptor/confidence head.
        mask = batch["mask"]
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=reprs["trunk_seqs"].dtype)

        # 3. Adapt representations (trainable)
        s, z = self.adaptor(
            trunk_seqs=reprs["trunk_seqs"],
            trunk_pair=reprs["trunk_pair"],
            local_latents=reprs["local_latents"],
            ca_coords=reprs["ca_coords"],
            decoder_seqs=reprs.get("decoder_seqs"),
            mask=mask,
        )

        # 4. Confidence prediction (frozen)
        outputs = self.confidence_head(s=s, z=z, mask=mask)
        return outputs

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable (adaptor) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """Total number of trainable parameters (adaptor only)."""
        return sum(p.numel() for p in self.trainable_parameters())

    def num_frozen_parameters(self) -> int:
        """Total number of frozen parameters (La-Proteina + confidence head)."""
        return sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
