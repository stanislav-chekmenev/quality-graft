"""Boltz confidence head wrapper for Quality-Graft.

Implements the custom pLDDT-only forward path from plans/architecture.md §6:

    adaptor outputs (s, z) -> pairformer -> linear heads (pLDDT / PDE / resolved)

The module instantiates the Boltz ``ConfidenceModule`` with the original
``imitate_trunk=True`` architecture (matching ``boltz1_conf.ckpt``) and then
loads checkpoint weights strictly from the ``confidence_module.*`` prefix.

Unlike the native Boltz forward pass, this wrapper:

- **Bypasses** Boltz token/atom input embedding, MSA stack, s_inputs
  projections, recycling, and diffusion conditioning — because Quality-Graft
  provides adapted single/pair representations from La-Proteina via the
  adaptor.
- **Bypasses** ``ConfidenceHeads.forward()`` entirely — aggregate/interface
  metrics (complex_plddt, iPDE, iPAE, pTM, ipTM) require Boltz-native features
  (``mol_type``, ``asym_id``, ``pred_distogram_logits``) unavailable here.
  Instead, the individual linear heads (``to_plddt_logits``, ``to_pde_logits``,
  ``to_resolved_logits``) are called directly.

The frozen MSA module (3.2M params) is instantiated for checkpoint compatibility
but never called.  See plans/architecture.md §6 for the full bypass rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from boltz.model.modules.confidence import ConfidenceModule


# Default MSA args matching boltz1_conf.ckpt architecture.
# The MSA module is instantiated for checkpoint weight compatibility but is
# never called in the Quality-Graft forward path.
_DEFAULT_MSA_ARGS: dict[str, Any] = {
    "msa_s": 64,
    "msa_blocks": 4,
    "msa_dropout": 0.15,
    "z_dropout": 0.25,
    "pairwise_head_width": 32,
    "pairwise_num_heads": 4,
    "postpone_outer_product": True,
    "activation_checkpointing": False,
    "offload_to_cpu": False,
}


class BoltzConfidenceHead(nn.Module):
    """Quality-Graft wrapper around Boltz ``ConfidenceModule``.

    All configuration is passed as plain Python dicts.  Hydra instantiation
    (via ``_target_``) happens at the top-level model assembly and converts
    OmegaConf nodes to plain dicts before reaching this constructor.

    Parameters
    ----------
    token_s : int
        Single representation dim.
    token_z : int
        Pair representation dim.
    pairformer_args : dict
        Boltz pairformer args.
    confidence_model_args : dict
        Confidence model args (num bins, feature toggles, head args).
    full_embedder_args : dict
        Boltz input embedder args (kept for checkpoint compatibility).
    msa_args : dict | None
        Boltz MSA module args (kept for checkpoint compatibility; the MSA
        module is instantiated but never called).  When ``None``, the
        module-level ``_DEFAULT_MSA_ARGS`` are used.
    imitate_trunk : bool
        Must stay ``True`` for boltz1 confidence checkpoint compatibility.
    ckpt_path : str
        Path to the ``boltz1_conf.ckpt`` checkpoint.
    ckpt_prefix : str
        Prefix for confidence-module keys in checkpoint state dict.
    device : str
        Device where the confidence module is placed.
    strict_loading : bool
        Require exact key match when loading weights.
    freeze : bool
        Freeze all confidence module parameters after loading.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        pairformer_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        full_embedder_args: dict[str, Any],
        ckpt_path: str,
        ckpt_prefix: str,
        device: str,
        msa_args: dict[str, Any] | None = None,
        compute_pae: bool = True,
        imitate_trunk: bool = True,
        strict_loading: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        self.token_s = token_s
        self.token_z = token_z
        self.device_name = device
        self.ckpt_path = ckpt_path
        self.ckpt_prefix = ckpt_prefix
        self.strict_loading = strict_loading

        if msa_args is None:
            msa_args = dict(_DEFAULT_MSA_ARGS)

        self.confidence_module = ConfidenceModule(
            token_s=token_s,
            token_z=token_z,
            compute_pae=compute_pae,
            imitate_trunk=imitate_trunk,
            pairformer_args=pairformer_args,
            full_embedder_args=full_embedder_args,
            msa_args=msa_args,
            **confidence_model_args,
        )
        self.confidence_module = self.confidence_module.to(device)

        self._load_confidence_weights(
            ckpt_path=ckpt_path,
            ckpt_prefix=ckpt_prefix,
            strict=strict_loading,
            device=device,
        )

        if freeze:
            self.confidence_module.requires_grad_(False)
            self.confidence_module.eval()

    def _load_confidence_weights(
        self,
        ckpt_path: str,
        ckpt_prefix: str,
        strict: bool,
        device: str,
    ) -> None:
        """Load confidence weights from checkpoint with strict key checking."""
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Confidence checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        confidence_state_dict = {
            key[len(ckpt_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(ckpt_prefix)
        }
        if not confidence_state_dict:
            raise RuntimeError(
                f"No keys with prefix '{ckpt_prefix}' found in checkpoint: {checkpoint_path}"
            )

        load_result = self.confidence_module.load_state_dict(
            confidence_state_dict,
            strict=strict,
        )

        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Confidence weight loading mismatch. "
                f"Missing: {missing[:5]} (total={len(missing)}), "
                f"Unexpected: {unexpected[:5]} (total={len(unexpected)})"
            )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        use_kernels: bool = False,
    ) -> dict[str, Tensor]:
        """Run the custom pLDDT-only confidence forward pass.

        This bypasses the full ``ConfidenceHeads.forward()`` and calls the
        individual linear heads directly.  Only the raw per-residue logits are
        returned — no aggregate/interface metrics (complex_plddt, iPDE, iPAE,
        pTM, ipTM) because those require Boltz-native features (``mol_type``,
        ``asym_id``, ``pred_distogram_logits``, etc.) that are unavailable in
        the Quality-Graft pipeline.

        Parameters
        ----------
        s : Tensor
            Adapted single representation ``[b, n, token_s]``.
        z : Tensor
            Adapted pair representation ``[b, n, n, token_z]``.  Already
            contains C-alpha distogram information from the adaptor.
        mask : Tensor
            Residue mask ``[b, n]`` (1 = valid, 0 = padding).
        use_kernels : bool
            Passed through to pairformer module.

        Returns
        -------
        dict[str, Tensor]
            ``plddt_logits``  ``[b, n, 50]``  — per-residue pLDDT bin logits.
            ``pde_logits``    ``[b, n, n, 64]`` — pairwise distance error logits.
            ``resolved_logits`` ``[b, n, 2]``  — per-residue resolved logits.
        """
        cm = self.confidence_module

        # Pairformer (distogram already in z from adaptor)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s, z = cm.pairformer_module(
            s,
            z,
            mask=mask,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )
        s = cm.final_s_norm(s)
        z = cm.final_z_norm(z)

        # Bypass ConfidenceHeads.forward() — call linear heads directly
        heads = cm.confidence_heads
        return {
            "plddt_logits": heads.to_plddt_logits(s),
            "pde_logits": heads.to_pde_logits(z + z.transpose(1, 2)),
            "resolved_logits": heads.to_resolved_logits(s),
        }
