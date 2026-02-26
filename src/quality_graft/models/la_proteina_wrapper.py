"""La-Proteina Wrapper for Quality-Graft.

Wraps the full La-Proteina pipeline: autoencoder encoder + trunk + optional decoder.
All components are frozen. Exposes intermediate representations (seqs, pair_rep,
local_latents) via replicated forward passes for downstream quality prediction.

Architecture reference: plans/architecture.md Section 5.1

Pipeline:
  1. Autoencoder encoder: all-atom coords + residue types -> z_latent [b,n,8]
  2. Construct flow matching batch at t=1.0 (clean sample)
  3. Trunk forward (replicated): -> seqs [b,n,768], pair_rep [b,n,n,256],
                                    local_latents_out [b,n,8], ca_out [b,n,3]
  4. Optional decoder forward (replicated): -> decoder_seqs [b,n,768]

Why replicate forward passes instead of hooks/subclassing?
  Both the trunk's and decoder's forward() methods return only their final output
  dicts. The intermediate seqs and pair_rep tensors are local variables. Replicating
  the forward pass in the wrapper is cleaner than monkey-patching or using hooks,
  and keeps the original La-Proteina code unmodified.
"""

import torch
import torch.nn as nn

from typing import Dict, Optional

from la_proteina.proteinfoundation.proteina import Proteina


class LaProteinaWrapper(nn.Module):
    """Wraps La-Proteina model components to extract intermediate representations.

    All sub-modules are **frozen** (no gradients). The wrapper replicates the
    forward passes of the trunk and (optionally) decoder to expose the
    intermediate ``seqs`` and ``pair_rep`` tensors that are normally hidden
    as local variables inside the original forward methods.

    Parameters
    ----------
    autoencoder_encoder : nn.Module
        The encoder part of La-Proteina's autoencoder (``EncoderTransformer``).
    trunk : nn.Module
        La-Proteina's trunk network (``LocalLatentsTransformer``).
    autoencoder_decoder : nn.Module, optional
        The decoder part of La-Proteina's autoencoder (``DecoderTransformer``).
        Required when ``use_decoder=True``.
    use_decoder : bool
        Whether to run the decoder forward pass and return ``decoder_seqs``.
        Set to ``False`` for Option A (trunk-only baseline), ``True`` for
        Option C (hybrid trunk + decoder).
    t_value : float
        Flow-matching time value for batch construction.
        ``t=1.0`` corresponds to the clean sample (no noise added).
    deterministic_encode : bool
        If ``True``, use the encoder's posterior mean instead of sampling
        from the latent distribution. Useful for deterministic evaluation.
        If ``False`` (default), use the reparameterised sample ``z_latent``,
        which matches the distribution the trunk was trained on.
    """

    def __init__(
        self,
        autoencoder_encoder: nn.Module,
        trunk: nn.Module,
        autoencoder_decoder: Optional[nn.Module] = None,
        use_decoder: bool = False,
        t_value: float = 1.0,
        deterministic_encode: bool = False,
    ):
        super().__init__()
        self.autoencoder_encoder = autoencoder_encoder
        self.trunk = trunk
        self.use_decoder = use_decoder
        self.t_value = t_value
        self.deterministic_encode = deterministic_encode

        if use_decoder:
            if autoencoder_decoder is None:
                raise ValueError(
                    "use_decoder=True requires autoencoder_decoder to be provided"
                )
            self.decoder = autoencoder_decoder

        # Freeze everything -- no gradients through La-Proteina
        self.requires_grad_(False)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_proteina_model(
        cls,
        proteina_model: "Proteina",  # noqa: F821
        use_decoder: bool = False,
        t_value: float = 1.0,
        deterministic_encode: bool = False,
    ) -> "LaProteinaWrapper":
        """Create a wrapper from a loaded ``Proteina`` LightningModule.

        Parameters
        ----------
        proteina_model : Proteina
            A loaded La-Proteina model instance.
        use_decoder : bool
            Whether to use the decoder (Option C).
        t_value : float
            Flow matching time value (1.0 = clean sample).
        deterministic_encode : bool
            Whether to use deterministic encoding (posterior mean).

        Returns
        -------
        LaProteinaWrapper
        """
        decoder = (
            proteina_model.autoencoder.decoder if use_decoder else None
        )
        return cls(
            autoencoder_encoder=proteina_model.autoencoder.encoder,
            trunk=proteina_model.nn,
            autoencoder_decoder=decoder,
            use_decoder=use_decoder,
            t_value=t_value,
            deterministic_encode=deterministic_encode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        proteina_ckpt_path: str,
        use_decoder: bool = False,
        t_value: float = 1.0,
        deterministic_encode: bool = False,
        device: str = "cpu",
        autoencoder_ckpt_path: Optional[str] = None,
    ) -> "LaProteinaWrapper":
        """Load a wrapper directly from a Proteina checkpoint.

        This handles importing the ``Proteina`` class, loading the checkpoint,
        and extracting the relevant sub-modules.

        Parameters
        ----------
        proteina_ckpt_path : str
            Path to the Proteina ``.ckpt`` file.
        use_decoder : bool
            Whether to use the decoder (Option C).
        t_value : float
            Flow matching time value.
        deterministic_encode : bool
            Whether to use deterministic encoding.
        device : str
            Device to load the checkpoint onto.
        autoencoder_ckpt_path : str, optional
            Override path for the autoencoder checkpoint. If ``None``, uses
            the path embedded in the Proteina checkpoint's config.

        Returns
        -------
        LaProteinaWrapper
        """

        kwargs = {}
        if autoencoder_ckpt_path is not None:
            kwargs["autoencoder_ckpt_path"] = autoencoder_ckpt_path

        proteina_model = Proteina.load_from_checkpoint(
            proteina_ckpt_path,
            map_location=device,
            strict=False,
            weights_only=False,
            **kwargs,
        )
        proteina_model.eval()

        return cls.from_proteina_model(
            proteina_model=proteina_model,
            use_decoder=use_decoder,
            t_value=t_value,
            deterministic_encode=deterministic_encode,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract La-Proteina intermediate representations from a protein structure.

        Parameters
        ----------
        batch : dict
            Data batch containing at minimum:

            - ``coords_nm`` : ``[b, n, 37, 3]`` all-atom coordinates in nanometres
            - ``coord_mask`` : ``[b, n, 37]`` atom-level boolean mask
            - ``mask`` or ``mask_dict`` : residue-level boolean mask
            - ``residue_type`` : ``[b, n]`` amino-acid types (integers 0–19)

        Returns
        -------
        dict
            - ``trunk_seqs``     : ``[b, n, 768]``   sequence repr after trunk layers
            - ``trunk_pair``     : ``[b, n, n, 256]`` pair repr after trunk layers
            - ``local_latents``  : ``[b, n, 8]``     predicted local latent variables
            - ``ca_coords``      : ``[b, n, 3]``     predicted CA coordinates
            - ``decoder_seqs``   : ``[b, n, 768]``   *(only if use_decoder=True)*
        """
        # Ensure mask formats expected by the encoder
        batch = self._ensure_mask(batch)

        # --- Step 1: Encode to get per-residue latents ---
        z_latent = self._encode(batch)  # [b, n, latent_dim]

        # --- Step 2: Construct flow matching batch at t ~= 1 ---
        ca_coords = batch["coords_nm"][:, :, 1, :]  # CA is atom index 1 → [b, n, 3]
        mask = self._get_mask(batch)  # [b, n]
        fm_batch = self._construct_fm_batch(batch, ca_coords, z_latent, mask)

        # --- Step 3: Trunk forward (replicated to expose intermediates) ---
        trunk_seqs, trunk_pair, local_latents_out, ca_out = self._trunk_forward(
            fm_batch
        )

        outputs = {
            "trunk_seqs": trunk_seqs,           # [b, n, 768]
            "trunk_pair": trunk_pair,           # [b, n, n, 256]
            "local_latents": local_latents_out, # [b, n, 8]
            "ca_coords": ca_out,                # [b, n, 3]
        }

        # --- Step 4: Optional decoder forward ---
        if self.use_decoder:
            decoder_seqs = self._decoder_forward(
                local_latents=local_latents_out,
                ca_coors_nm=ca_out,
                mask=mask,
            )
            outputs["decoder_seqs"] = decoder_seqs  # [b, n, 768]

        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_mask(self, batch: Dict) -> Dict:
        """Ensure the batch has masks in the formats expected by sub-modules.

        The autoencoder encoder reads ``batch["mask_dict"]["coords"][..., 0, 0]``
        to get a ``[b, n]`` boolean mask.  If the batch only has a flat ``mask``
        or ``coord_mask``, this method constructs the required ``mask_dict``.
        """
        # Derive flat mask if not present
        if "mask" not in batch:
            if "mask_dict" in batch:
                batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            elif "coord_mask" in batch:
                batch["mask"] = batch["coord_mask"].any(dim=-1)
            else:
                raise ValueError(
                    "Batch must contain 'mask', 'mask_dict', or 'coord_mask'"
                )

        # Build mask_dict if not present (encoder expects this structure)
        if "mask_dict" not in batch:
            mask = batch["mask"]  # [b, n]
            # Expand to [b, n, 37, 3] so that [..., 0, 0] recovers [b, n]
            mask_expanded = mask[:, :, None, None].expand(-1, -1, 37, 3)
            batch["mask_dict"] = {
                "coords": mask_expanded,
                "residue_type": mask,
            }

        return batch

    def _get_mask(self, batch: Dict) -> torch.Tensor:
        """Extract the ``[b, n]`` residue mask from the batch."""
        if "mask" in batch:
            return batch["mask"]
        if "mask_dict" in batch:
            return batch["mask_dict"]["coords"][..., 0, 0]
        raise ValueError("Cannot extract mask from batch")

    def _encode(self, batch: Dict) -> torch.Tensor:
        """Run the autoencoder encoder to get per-residue latent variables.

        Parameters
        ----------
        batch : dict
            Full data batch (must contain ``mask_dict``, ``coords_nm``,
            ``coord_mask``, ``residue_type``, etc.).

        Returns
        -------
        torch.Tensor
            ``z_latent`` of shape ``[b, n, latent_dim]``.
        """
        encoded = self.autoencoder_encoder(batch)
        if self.deterministic_encode:
            return encoded["mean"]  # [b, n, latent_dim]
        return encoded["z_latent"]  # [b, n, latent_dim]

    def _construct_fm_batch(
        self,
        batch: Dict,
        ca_coords: torch.Tensor,
        z_latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict:
        """Construct a flow-matching batch at time *t* for the trunk.

        At ``t = 1.0`` the noisy sample ``x_t`` equals the clean data, which
        is appropriate for extracting representations from a known structure.
        A shallow copy of the original batch is made so that feature factories
        (index embeddings, pair distances, etc.) have access to all auxiliary
        data without duplication.

        Parameters
        ----------
        batch : dict
            Original data batch.
        ca_coords : torch.Tensor
            CA coordinates ``[b, n, 3]``.
        z_latent : torch.Tensor
            Latent variables ``[b, n, latent_dim]``.
        mask : torch.Tensor
            Residue mask ``[b, n]``.

        Returns
        -------
        dict
            Augmented batch with ``x_t`` and ``t`` keys.
        """
        b = ca_coords.shape[0]
        device = ca_coords.device
        dtype = ca_coords.dtype

        fm_batch = dict(batch)  # shallow copy — does NOT duplicate tensors
        fm_batch["mask"] = mask
        fm_batch["x_t"] = {
            "bb_ca": ca_coords,        # [b, n, 3]
            "local_latents": z_latent,  # [b, n, latent_dim]
        }
        fm_batch["t"] = {
            "bb_ca": torch.full((b,), self.t_value, device=device, dtype=dtype),
            "local_latents": torch.full(
                (b,), self.t_value, device=device, dtype=dtype
            ),
        }
        # Explicitly disable optional conditioning features so the trunk
        # feature factories fall back to zeros for these slots.
        fm_batch["use_ca_coors_nm_feature"] = False
        fm_batch["use_residue_type_feature"] = False

        return fm_batch

    def _trunk_forward(
        self, fm_batch: Dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replicated trunk forward that exposes intermediate representations.

        Instead of calling ``self.trunk(fm_batch)`` — which returns only the
        ``nn_out`` dict with output-head predictions — we replicate the
        internal forward logic of ``LocalLatentsTransformer`` to capture
        ``seqs`` and ``pair_rep`` **after** the transformer layers and
        **before** the output-projection heads.

        Parameters
        ----------
        fm_batch : dict
            Flow-matching batch with ``x_t``, ``t``, ``mask``, etc.

        Returns
        -------
        seqs : torch.Tensor
            ``[b, n, 768]`` — sequence representations after all trunk layers.
        pair_rep : torch.Tensor
            ``[b, n, n, 256]`` — pair representations after all trunk layers.
        local_latents_out : torch.Tensor
            ``[b, n, 8]`` — predicted local latent variables.
        ca_out : torch.Tensor
            ``[b, n, 3]`` — predicted CA coordinates.
        """
        mask = fm_batch["mask"]  # [b, n]

        # --- Conditioning ---
        c = self.trunk.cond_factory(fm_batch)  # [b, n, dim_cond]
        c = self.trunk.transition_c_2(
            self.trunk.transition_c_1(c, mask), mask
        )  # [b, n, dim_cond]

        # --- Initial representations ---
        seqs = self.trunk.init_repr_factory(fm_batch) * mask[..., None]  # [b, n, 768]
        pair_rep = self.trunk.pair_repr_builder(fm_batch)  # [b, n, n, 256]

        # --- Run trunk transformer layers ---
        for i in range(self.trunk.nlayers):
            seqs = self.trunk.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, 768]

            if self.trunk.update_pair_repr:
                if i < self.trunk.nlayers - 1:
                    if self.trunk.pair_update_layers[i] is not None:
                        pair_rep = self.trunk.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, 256]

        # --- Output heads ---
        local_latents_out = (
            self.trunk.local_latents_linear(seqs) * mask[..., None]
        )  # [b, n, 8]
        ca_out = self.trunk.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        return seqs, pair_rep, local_latents_out, ca_out

    def _decoder_forward(
        self,
        local_latents: torch.Tensor,
        ca_coors_nm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replicated decoder forward that exposes intermediate ``seqs``.

        We replicate the ``DecoderTransformer.forward()`` logic to capture
        ``seqs`` after the transformer layers, **before** the output heads
        (``logit_linear``, ``struct_linear``).

        Parameters
        ----------
        local_latents : torch.Tensor
            ``[b, n, 8]`` — local latent variables from trunk output.
        ca_coors_nm : torch.Tensor
            ``[b, n, 3]`` — CA coordinates from trunk output.
        mask : torch.Tensor
            ``[b, n]`` — residue mask.

        Returns
        -------
        torch.Tensor
            ``[b, n, 768]`` — decoder sequence representations enriched with
            latent and CA coordinate information.
        """
        decoder_input = {
            "z_latent": local_latents,
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "mask": mask,
        }

        # --- Conditioning ---
        c = self.decoder.cond_factory(decoder_input)
        c = self.decoder.transition_c_2(
            self.decoder.transition_c_1(c, mask), mask
        )

        # --- Initial representations ---
        seqs = (
            self.decoder.init_repr_factory(decoder_input) * mask[..., None]
        )  # [b, n, 768]
        pair_rep = self.decoder.pair_rep_factory(decoder_input)  # [b, n, n, 256]

        # --- Run decoder transformer layers ---
        for i in range(self.decoder.nlayers):
            seqs = self.decoder.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, 768]

            if self.decoder.update_pair_repr:
                if i < self.decoder.nlayers - 1:
                    if self.decoder.pair_update_layers[i] is not None:
                        pair_rep = self.decoder.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, 256]

        return seqs  # [b, n, 768]
