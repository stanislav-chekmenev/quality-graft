"""Adaptor module for Quality-Graft.

Adapts La-Proteina representations (seqs, pair_rep, local_latents) to the
Boltz1 confidence head input space (single repr ``s`` and pair repr ``z``).

The adaptor is parameterised by ``source_mode`` to support smooth transition
from Option A (trunk-only) to Option C (hybrid trunk + decoder).  It uses
configurable self-attention layers (0–2) in the target Boltz1 dimension space
to refine the projected representations before they enter the frozen
pairformer.

Architecture reference: plans/architecture.md Section 5.2

Dimension mappings
------------------
+---------------------+-----------+------------+-----------------------------------+
| Mapping             | Input Dim | Output Dim | Notes                             |
+---------------------+-----------+------------+-----------------------------------+
| Single projection   | 776       | 384        | seqs (768) + local_latents (8)    |
| Pair projection     | 256       | 128        | Trunk pair_rep (both modes)       |
| Decoder fusion (C)  | 768       | 768        | Gated addition into trunk seqs    |
+---------------------+-----------+------------+-----------------------------------+

Attention blocks
----------------
Each ``AdaptorAttentionBlock`` operates in the **target** Boltz1 space and
contains:

- Pair-biased self-attention on single repr ``s`` (uses pair repr ``z`` as
  bias), matching the ``AttentionPairBias`` pattern used in Boltz1's
  pairformer.
- SwiGLU transition FFN on single repr ``s``.
- SwiGLU transition FFN on pair repr ``z``.

Both the attention output projection (``proj_o``) and the transition output
(``fc3``) are **zero-initialised**, so the attention blocks start as
near-identity transforms.  This means:

- ``n_attn_layers=0``: pure linear projection (Phase 1a).
- ``n_attn_layers=1–2``: attention-augmented (Phase 1b), with graceful
  degradation to linear at init.

Transition from Option A to Option C
-------------------------------------
1. Train with ``source_mode="trunk"`` – ``single_proj`` and ``pair_proj``
   (plus optional attention blocks) learn the mapping.
2. Switch to ``source_mode="hybrid"`` – ``decoder_fusion`` is added with
   zero-initialised weights.
3. Load Option A weights via ``strict=False``.
4. Fine-tune – ``decoder_fusion`` gradually learns to incorporate decoder
   information while existing projections can be fine-tuned or frozen.
"""

import torch
import torch.nn as nn

from loguru import logger

from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.transition import Transition


class AdaptorAttentionBlock(nn.Module):
    """Single attention + lightweight MLP block for the adaptor.

    Operates in the target Boltz1 dimension space.  All output projections
    are zero-initialised so this block starts as near-identity and gradually
    learns non-linear refinements through training.

    Parameters
    ----------
    s_dim : int
        Single representation dimension (target space, e.g. 384).
    z_dim : int
        Pair representation dimension (target space, e.g. 128).
    num_heads : int
        Number of attention heads for pair-biased self-attention.
    s_ff_factor : int
        Expansion factor for single repr transition FFN hidden dim.
    z_ff_factor : int
        Expansion factor for pair repr transition FFN hidden dim.
    """

    def __init__(
        self,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
        s_ff_factor: int = 2,  # kept for API compatibility
        z_ff_factor: int = 2,  # kept for API compatibility
    ):
        super().__init__()

        # --- Single: pair-biased self-attention ---
        # Uses Boltz1's AttentionPairBias which includes:
        #   - LayerNorm on s (initial_norm=True)
        #   - Q/K/V projections + gating
        #   - LayerNorm + linear projection on z for pair bias
        #   - Zero-initialised output projection (proj_o)

        _, _ = s_ff_factor, z_ff_factor  # Not used in this block, but kept for API compatibility

        # Attention block (single repr self-attention with pair bias)
        self.attn = AttentionPairBias(
            c_s=s_dim,
            c_z=z_dim,
            num_heads=num_heads,
            initial_norm=True,
        )
        self.silu = nn.SiLU()

        # Simple 1-layer MLPs (LayerNorm -> Linear), replacing Transition
        self.s_mlp = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim, bias=False),
        )
        self.z_mlp = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, z_dim, bias=False),
        )

        # Start as near-identity residual blocks
        nn.init.zeros_(self.s_mlp[1].weight)
        nn.init.zeros_(self.z_mlp[1].weight)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the attention block.

        Parameters
        ----------
        s : torch.Tensor
            Single representation ``[b, n, s_dim]``.
        z : torch.Tensor
            Pair representation ``[b, n, n, z_dim]``.
        mask : torch.Tensor
            Residue mask ``[b, n]`` (1 = valid, 0 = padding).

        Returns
        -------
        s : torch.Tensor
            Refined single representation ``[b, n, s_dim]``.
        z : torch.Tensor
            Refined pair representation ``[b, n, n, z_dim]``.
        """
        # Pair-biased self-attention on single repr (residual)
        s = s + self.attn(s=s, z=z, mask=mask)

        # Simple 1-layer MLP refinements (residual)
        s = s + self.silu(self.s_mlp(s))
        z = z + self.silu(self.z_mlp(z))

        # Re-apply mask
        s = s * mask[..., None]
        z = z * mask[:, :, None, None] * mask[:, None, :, None]

        return s, z


class AdaptorModule(nn.Module):
    """Adapts La-Proteina representations to Boltz1 confidence head input space.

    Supports two modes:
      - ``"trunk"``: Uses trunk seqs + local_latents (Option A baseline).
      - ``"hybrid"``: Fuses trunk seqs with decoder seqs, plus local_latents
        (Option C).

    In both modes the ``single_proj`` and ``pair_proj`` layers are identical.
    Only ``decoder_fusion`` is added in ``"hybrid"`` mode.

    Parameters
    ----------
    source_mode : str
        ``"trunk"`` or ``"hybrid"``.
    trunk_dim : int
        La-Proteina trunk single repr dimension (768).
    pair_dim : int
        La-Proteina trunk pair repr dimension (256).
    latent_dim : int
        Local latent variable dimension (8).
    target_s_dim : int
        Boltz1 single repr dimension (384).
    target_z_dim : int
        Boltz1 pair repr dimension (128).
    n_attn_layers : int
        Number of self-attention refinement layers (0 = linear only,
        1–2 = attention-augmented).
    num_heads : int
        Number of attention heads per block.
    s_ff_factor : int
        Expansion factor for single repr transition FFN.
    z_ff_factor : int
        Expansion factor for pair repr transition FFN.
    """

    def __init__(
        self,
        source_mode: str = "trunk",
        trunk_dim: int = 768,
        pair_dim: int = 256,
        latent_dim: int = 8,
        target_s_dim: int = 384,
        target_z_dim: int = 128,
        n_attn_layers: int = 1,
        num_heads: int = 16,
        s_ff_factor: int = 2,
        z_ff_factor: int = 2,
    ):
        super().__init__()
        self.source_mode = source_mode
        self.n_attn_layers = n_attn_layers

        # Validate the source_mode
        if source_mode not in ("trunk", "hybrid"):
            raise ValueError(
                f"source_mode must be 'trunk' or 'hybrid', got {source_mode!r}"
            )
        
        # --- Decoder fusion gate (Option C only) ---
        if source_mode == "hybrid":
            self.decoder_fusion = nn.Sequential(
                nn.LayerNorm(trunk_dim),
                nn.Linear(trunk_dim, trunk_dim, bias=False),
            )
            # Zero-initialise so that at init, hybrid == trunk behaviour
            nn.init.zeros_(self.decoder_fusion[1].weight)

        # --- Single adaptor: seqs (possibly fused) + local_latents -> s ---
        single_input_dim = trunk_dim + latent_dim  # 768 + 8 = 776
        self.single_proj = nn.Sequential(
            nn.LayerNorm(single_input_dim),
            nn.Linear(single_input_dim, target_s_dim, bias=False),
        )

        # --- Pair adaptor: trunk pair_rep -> z ---
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, target_z_dim, bias=False),
        )

        # --- Self-attention refinement blocks ---
        if n_attn_layers > 0:
            self.attn_blocks = nn.ModuleList(
                [
                    AdaptorAttentionBlock(
                        s_dim=target_s_dim,
                        z_dim=target_z_dim,
                        num_heads=num_heads,
                        s_ff_factor=s_ff_factor,
                        z_ff_factor=z_ff_factor,
                    )
                    for _ in range(n_attn_layers)
                ]
            )

    def forward(
        self,
        trunk_seqs: torch.Tensor,
        trunk_pair: torch.Tensor,
        local_latents: torch.Tensor,
        decoder_seqs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt La-Proteina representations to Boltz1 input space.

        Parameters
        ----------
        trunk_seqs : torch.Tensor
            Trunk single representation ``[b, n, 768]``.
        trunk_pair : torch.Tensor
            Trunk pair representation ``[b, n, n, 256]``.
        local_latents : torch.Tensor
            Local latent variables ``[b, n, 8]``.
        decoder_seqs : torch.Tensor, optional
            Decoder single representation ``[b, n, 768]``.
            Only used when ``source_mode="hybrid"``.
        mask : torch.Tensor, optional
            Residue mask ``[b, n]`` (1 = valid, 0 = padding).
            Required when ``n_attn_layers > 0``.

        Returns
        -------
        s : torch.Tensor
            Adapted single representation ``[b, n, 384]``.
        z : torch.Tensor
            Adapted pair representation ``[b, n, n, 128]``.
        """
        # --- Single representation ---
        if self.source_mode == "hybrid" and decoder_seqs is not None:
            # Fuse decoder signal into trunk seqs via gated addition
            fused_seqs = trunk_seqs + self.decoder_fusion(decoder_seqs)
        elif self.source_mode == "hybrid" and decoder_seqs is None:
            logger.warning(
                "source_mode='hybrid' but decoder_seqs is None. "
                "Falling back to trunk-only input for single representation."
            )
            fused_seqs = trunk_seqs
        else:
            fused_seqs = trunk_seqs

        single_in = torch.cat([fused_seqs, local_latents], dim=-1)  # [b, n, 776]
        s = self.single_proj(single_in)  # [b, n, 384]

        # --- Pair representation ---
        z = self.pair_proj(trunk_pair)  # [b, n, n, 128]

        # --- Self-attention refinement ---
        if self.n_attn_layers > 0:
            if mask is None:
                # Default: all residues valid
                mask = torch.ones(
                    s.shape[:2], dtype=s.dtype, device=s.device
                )
            for block in self.attn_blocks:
                s, z = block(s, z, mask)

        return s, z
