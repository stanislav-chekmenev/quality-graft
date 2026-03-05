"""Full Quality-Graft assembled model.

Wires together the three core components:

    La-Proteina (frozen) → Adaptor (trainable) → Boltz1 Confidence Head (frozen)

The model accepts a protein-structure batch (coordinates, masks, residue types)
and produces per-residue quality predictions (pLDDT, PDE, resolved logits).

Architecture reference: plans/architecture.md Section 5.4

Sub-module responsibilities
---------------------------
- **La-Proteina wrapper** (frozen): extracts trunk_seqs, trunk_pair,
  local_latents, ca_coords, and optionally decoder_seqs from all-atom
  coordinates via the autoencoder encoder + flow matcher + trunk + optional
  decoder.
- **Adaptor** (trainable): projects La-Proteina representations into Boltz1
  dimension space (single 776→384, pair 256→128) with optional self-attention
  refinement and C-alpha distogram injection.
- **Confidence head** (frozen): runs the 48-block pairformer stack and
  produces pLDDT/PDE/resolved logits via the linear prediction heads.

Two construction patterns are supported:

1. **Dependency injection** — pass pre-built sub-modules to ``__init__``.
2. **Forward from representations** — call :meth:`forward_from_representations`
   with pre-extracted La-Proteina features, bypassing the wrapper entirely
   (useful for testing or pre-computed feature pipelines).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.confidence_head import BoltzConfidenceHead
from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper


class QualityGraft(nn.Module):
    """Full Quality-Graft model.

    La-Proteina (frozen) → Adaptor (trainable) → Boltz1 Confidence (frozen) → pLDDT

    Parameters
    ----------
    la_proteina : LaProteinaWrapper
        Frozen La-Proteina wrapper that extracts intermediate representations.
    adaptor : AdaptorModule
        Trainable adaptor that projects La-Proteina representations into
        Boltz1 dimension space.
    confidence_head : BoltzConfidenceHead
        Frozen Boltz1 confidence head with pairformer + prediction heads.
    """

    def __init__(
        self,
        la_proteina: LaProteinaWrapper,
        adaptor: AdaptorModule,
        confidence_head: BoltzConfidenceHead,
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
