"""Boltz confidence head wrapper for Quality-Graft.

Implements the custom forward path from plans/architecture.md todo item 8:

    adaptor outputs (s, z) -> distogram update -> pairformer -> confidence heads

The module instantiates the Boltz ``ConfidenceModule`` with the original
``imitate_trunk=True`` architecture (matching ``boltz1_conf.ckpt``) and then
loads checkpoint weights strictly from the ``confidence_module.*`` prefix.

Unlike the native Boltz forward pass, this wrapper bypasses Boltz token/atom
input embedding and MSA stack because Quality-Graft already provides adapted
single/pair representations from La-Proteina via the adaptor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from boltz.data import const
from boltz.model.modules.confidence import ConfidenceModule


def _to_plain_dict(cfg: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Convert nested config objects to a plain Python dict."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)  # type: ignore[return-value]


class BoltzConfidenceHead(nn.Module):
    """Quality-Graft wrapper around Boltz ``ConfidenceModule``.

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
    msa_args : dict
        Boltz MSA args.
    compute_pae : bool
        Whether to keep PAE head active.
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
        pairformer_args: dict[str, Any] | DictConfig,
        confidence_model_args: dict[str, Any] | DictConfig,
        full_embedder_args: dict[str, Any] | DictConfig,
        msa_args: dict[str, Any] | DictConfig,
        ckpt_path: str,
        ckpt_prefix: str,
        device: str,
        compute_pae: bool = True,
        imitate_trunk: bool = True,
        strict_loading: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        pairformer_args_dict = _to_plain_dict(pairformer_args)
        confidence_model_args_dict = _to_plain_dict(confidence_model_args)
        full_embedder_args_dict = _to_plain_dict(full_embedder_args)
        msa_args_dict = _to_plain_dict(msa_args)

        self.token_s = token_s
        self.token_z = token_z
        self.device_name = device
        self.ckpt_path = ckpt_path
        self.ckpt_prefix = ckpt_prefix
        self.strict_loading = strict_loading

        self.confidence_module = ConfidenceModule(
            token_s=token_s,
            token_z=token_z,
            compute_pae=compute_pae,
            imitate_trunk=imitate_trunk,
            pairformer_args=pairformer_args_dict,
            full_embedder_args=full_embedder_args_dict,
            msa_args=msa_args_dict,
            **confidence_model_args_dict,
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

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> "BoltzConfidenceHead":
        """Instantiate from a Hydra/OmegaConf config node.

        The config must include ``_target_: quality_graft.models.confidence_head.BoltzConfidenceHead``.
        """
        instance = instantiate(cfg)
        if not isinstance(instance, cls):
            raise TypeError(
                "Hydra config did not instantiate BoltzConfidenceHead. "
                f"Got: {type(instance)!r}"
            )
        return instance

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

    def _build_msa_single_repr_inputs(
        self,
        s: Tensor,
        msa_aa_tokens: Tensor | None,
    ) -> Tensor:
        """Build ``s_inputs`` expected by Boltz ``MSAModule`` from adapted ``s``.

        Uses adapted single representations as the atom-derived component and
        appends token/profile/deletion/pocket channels in the same layout as
        Boltz ``InputEmbedder`` output.
        """
        batch_size, n_tokens, _ = s.shape
        num_tokens = const.num_tokens
        pocket_dim = len(const.pocket_contact_info)

        if msa_aa_tokens is None:
            res_type = torch.zeros(
                batch_size,
                n_tokens,
                num_tokens,
                dtype=s.dtype,
                device=s.device,
            )
            profile = torch.zeros_like(res_type)
        else:
            if msa_aa_tokens.shape != (batch_size, n_tokens):
                raise ValueError(
                    "msa_aa_tokens must have shape [b, n] matching s. "
                    f"Got {tuple(msa_aa_tokens.shape)} vs {(batch_size, n_tokens)}"
                )
            msa_aa_tokens = msa_aa_tokens.long()
            res_type = nn.functional.one_hot(msa_aa_tokens, num_classes=num_tokens).to(s.dtype)
            profile = res_type

        deletion_mean = torch.zeros(
            batch_size,
            n_tokens,
            1,
            dtype=s.dtype,
            device=s.device,
        )
        pocket_feature = torch.zeros(
            batch_size,
            n_tokens,
            pocket_dim,
            dtype=s.dtype,
            device=s.device,
        )
        return torch.cat([s, res_type, profile, deletion_mean, pocket_feature], dim=-1)

    def _run_optional_msa(
        self,
        s: Tensor,
        z: Tensor,
        msa_feats: dict[str, Tensor] | None,
        msa_aa_tokens: Tensor | None,
        use_kernels: bool,
    ) -> Tensor:
        """Apply Boltz MSA module when MSA features are provided."""
        if msa_feats is None:
            return z

        required_keys = {"msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask", "token_pad_mask"}
        missing = required_keys.difference(msa_feats.keys())
        if missing:
            raise ValueError(f"msa_feats missing required keys: {sorted(missing)}")

        s_inputs = self._build_msa_single_repr_inputs(s, msa_aa_tokens)
        return z + self.confidence_module.msa_module(
            z=z,
            emb=s_inputs,
            feats=msa_feats,
            use_kernels=use_kernels,
        )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        x_pred: Tensor,
        feats: dict[str, Tensor],
        pred_distogram_logits: Tensor,
        multiplicity: int = 1,
        s_diffusion: Tensor | None = None,
        msa_feats: dict[str, Tensor] | None = None,
        msa_aa_tokens: Tensor | None = None,
        use_kernels: bool = False,
    ) -> dict[str, Tensor]:
        """Run the custom confidence forward pass.

        Parameters
        ----------
        s : Tensor
            Adapted single representation ``[b, n, token_s]``.
        z : Tensor
            Adapted pair representation ``[b, n, n, token_z]``.
        x_pred : Tensor
            Predicted atom coordinates ``[b, n_atoms, 3]``.
        feats : dict[str, Tensor]
            Features required by Boltz confidence heads (e.g.
            ``token_to_rep_atom``, ``token_pad_mask``, ``mol_type``, ``asym_id``,
            ``atom_to_token``, ``atom_pad_mask``, ``frames_idx``).
        pred_distogram_logits : Tensor
            Distogram logits used by confidence metrics.
        multiplicity : int
            Diffusion multiplicity, defaults to 1.
        s_diffusion : Tensor | None
            Optional diffusion single embeddings.
        msa_feats : dict[str, Tensor] | None
            Optional MSA feature dict. If ``None`` (default), the MSA module is
            skipped and behavior matches the current no-MSA path.
        msa_aa_tokens : Tensor | None
            Optional amino-acid token ids ``[b, n]`` used to build MSA single
            input channels (res_type/profile) when ``msa_feats`` is provided.
        use_kernels : bool
            Passed through to pairformer module.

        Returns
        -------
        dict[str, Tensor]
            Confidence outputs from Boltz ``ConfidenceHeads``.
        """
        confidence_module = self.confidence_module

        s = s.repeat_interleave(multiplicity, 0)
        z = z.repeat_interleave(multiplicity, 0)

        if confidence_module.use_s_diffusion:
            if s_diffusion is None:
                raise ValueError("s_diffusion must be provided when use_s_diffusion=True")
            s_diffusion = confidence_module.s_diffusion_norm(s_diffusion)
            s = s + confidence_module.s_diffusion_to_s(s_diffusion)

        token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > confidence_module.boundaries).sum(dim=-1).long()
        distogram = confidence_module.dist_bin_pairwise_embed(distogram)
        z = z + distogram
        z = self._run_optional_msa(
            s=s,
            z=z,
            msa_feats=msa_feats,
            msa_aa_tokens=msa_aa_tokens,
            use_kernels=use_kernels,
        )

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s, z = confidence_module.pairformer_module(
            s,
            z,
            mask=mask,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )
        s = confidence_module.final_s_norm(s)
        z = confidence_module.final_z_norm(z)

        return confidence_module.confidence_heads(
            s=s,
            z=z,
            x_pred=x_pred,
            d=d,
            feats=feats,
            multiplicity=multiplicity,
            pred_distogram_logits=pred_distogram_logits,
        )