"""Student confidence head for Quality-Graft distillation.

Replaces the frozen 48-block Boltz ``BoltzConfidenceHead`` with a smaller,
fully trainable student network.  The student reuses Boltz's
``PairformerModule`` building blocks (triangle multiplication, triangle
attention, pair-biased attention, transition FFNs) but with far fewer blocks
(typically 4-5 instead of 48), giving ~10-15M trainable parameters.

Architecture::

    s [b, n, 384], z [b, n, n, 128]
      → PairformerModule (N blocks, trainable)
      → LayerNorm(s), LayerNorm(z)
      → Linear heads: pLDDT (s→50), PDE (z→64), resolved (s→2)

The forward interface matches ``BoltzConfidenceHead.forward`` so it can be
used as a drop-in replacement in ``QualityGraft``.
"""

from __future__ import annotations

from torch import Tensor, nn
from boltz.model.modules.trunk import PairformerModule


class StudentConfidenceHead(nn.Module):
    """Trainable student confidence head for distillation.

    Parameters
    ----------
    token_s : int
        Single representation dimension (default 384).
    token_z : int
        Pair representation dimension (default 128).
    num_blocks : int
        Number of pairformer layers (default 4).
    num_heads : int
        Attention heads for pair-biased self-attention (default 16).
    dropout : float
        Dropout rate for triangle ops and attention (default 0.2).
    pairwise_head_width : int
        Head width for triangle attention (default 32).
    pairwise_num_heads : int
        Number of heads for triangle attention (default 4).
    num_plddt_bins : int
        Number of pLDDT output bins (default 50).
    num_pde_bins : int
        Number of PDE output bins (default 64).
    predict_pde : bool
        Whether to predict PDE logits (multi-task regularization).
    predict_resolved : bool
        Whether to predict resolved logits (multi-task regularization).
    """

    def __init__(
        self,
        token_s: int = 384,
        token_z: int = 128,
        num_blocks: int = 4,
        num_heads: int = 16,
        dropout: float = 0.2,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        num_plddt_bins: int = 50,
        num_pde_bins: int = 64,
        predict_pde: bool = True,
        predict_resolved: bool = True,
    ) -> None:
        super().__init__()

        self.predict_pde = predict_pde
        self.predict_resolved = predict_resolved

        # Pairformer stack (reuses Boltz building blocks)
        self.pairformer = PairformerModule(
            token_s=token_s,
            token_z=token_z,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            activation_checkpointing=False,
        )

        # Final layer norms
        self.final_s_norm = nn.LayerNorm(token_s)
        self.final_z_norm = nn.LayerNorm(token_z)

        # Prediction heads
        self.to_plddt_logits = nn.Linear(token_s, num_plddt_bins)

        if predict_pde:
            self.to_pde_logits = nn.Linear(token_z, num_pde_bins)

        if predict_resolved:
            self.to_resolved_logits = nn.Linear(token_s, 2)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        use_kernels: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass matching BoltzConfidenceHead interface.

        Parameters
        ----------
        s : Tensor
            Single representation ``[b, n, token_s]``.
        z : Tensor
            Pair representation ``[b, n, n, token_z]``.
        mask : Tensor
            Residue mask ``[b, n]`` (1=valid, 0=padding).
        use_kernels : bool
            Passed to pairformer.

        Returns
        -------
        dict[str, Tensor]
            ``plddt_logits``    — ``[b, n, num_plddt_bins]``
            ``pde_logits``      — ``[b, n, n, num_pde_bins]`` (if predict_pde)
            ``resolved_logits`` — ``[b, n, 2]`` (if predict_resolved)
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s, z = self.pairformer(
            s, z,
            mask=mask,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )

        s = self.final_s_norm(s)
        z = self.final_z_norm(z)

        outputs: dict[str, Tensor] = {
            "plddt_logits": self.to_plddt_logits(s),
        }

        if self.predict_pde:
            outputs["pde_logits"] = self.to_pde_logits(z + z.transpose(1, 2))

        if self.predict_resolved:
            outputs["resolved_logits"] = self.to_resolved_logits(s)

        return outputs
