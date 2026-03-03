"""Unit tests for the AdaptorModule and AdaptorAttentionBlock.

Tests cover:
  - Output shape correctness for trunk and hybrid modes
  - Zero-initialisation of attention blocks (near-identity at init)
  - Zero-initialisation of decoder fusion gate (hybrid == trunk at init)
  - Mask propagation (padding zeroed out)
  - Configurable attention depth (0, 1, 2 layers)
  - Parameter counts and gradient flow
"""

import pytest
import torch

from quality_graft.models.adaptor import AdaptorAttentionBlock, AdaptorModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_dims():
    """Standard batch / sequence dimensions for testing."""
    return {"b": 2, "n": 10}


@pytest.fixture
def default_dims():
    """Default La-Proteina -> Boltz1 dimension mapping."""
    return {
        "trunk_dim": 768,
        "pair_dim": 256,
        "latent_dim": 8,
        "target_s_dim": 384,
        "target_z_dim": 128,
    }


@pytest.fixture
def trunk_inputs(batch_dims, default_dims):
    """Random input tensors mimicking La-Proteina trunk outputs."""
    b, n = batch_dims["b"], batch_dims["n"]
    return {
        "trunk_seqs": torch.randn(b, n, default_dims["trunk_dim"]),
        "trunk_pair": torch.randn(b, n, n, default_dims["pair_dim"]),
        "local_latents": torch.randn(b, n, default_dims["latent_dim"]),
        "ca_coords": torch.randn(b, n, 3),
        "mask": torch.ones(b, n),
    }


# ---------------------------------------------------------------------------
# AdaptorAttentionBlock tests
# ---------------------------------------------------------------------------


class TestAdaptorAttentionBlock:
    """Tests for the individual attention block."""

    def test_output_shapes(self, batch_dims):
        """Block should preserve s and z shapes."""
        b, n = batch_dims["b"], batch_dims["n"]
        s_dim, z_dim = 384, 128

        block = AdaptorAttentionBlock(s_dim=s_dim, z_dim=z_dim, num_heads=16)
        s = torch.randn(b, n, s_dim)
        z = torch.randn(b, n, n, z_dim)
        mask = torch.ones(b, n)

        s_out, z_out = block(s, z, mask)

        assert s_out.shape == (b, n, s_dim)
        assert z_out.shape == (b, n, n, z_dim)

    def test_zero_init_near_identity(self, batch_dims):
        """At initialisation the block should be near-identity.

        Both Boltz1's AttentionPairBias (proj_o zero-init) and Transition
        (fc3 zero-init) start with zero output projections, so the residual
        connections make the block near-identity.
        """
        b, n = batch_dims["b"], batch_dims["n"]
        s_dim, z_dim = 384, 128

        block = AdaptorAttentionBlock(s_dim=s_dim, z_dim=z_dim, num_heads=16)
        s = torch.randn(b, n, s_dim)
        z = torch.randn(b, n, n, z_dim)
        mask = torch.ones(b, n)

        with torch.no_grad():
            s_out, z_out = block(s, z, mask)

        # The attention proj_o is zero-initialised, so attn output is 0
        # The transition fc3 is zero-initialised, so transition output is 0
        # Therefore s_out ≈ s and z_out ≈ z (up to masking)
        torch.testing.assert_close(s_out, s, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_out, z, atol=1e-5, rtol=1e-5)

    def test_mask_zeroes_padding(self, batch_dims):
        """Masked-out positions should be zeroed."""
        b, n = batch_dims["b"], batch_dims["n"]
        s_dim, z_dim = 384, 128

        block = AdaptorAttentionBlock(s_dim=s_dim, z_dim=z_dim, num_heads=16)
        s = torch.randn(b, n, s_dim)
        z = torch.randn(b, n, n, z_dim)

        # Mask out last 3 residues
        mask = torch.ones(b, n)
        mask[:, -3:] = 0.0

        with torch.no_grad():
            s_out, z_out = block(s, z, mask)

        # Padded positions in s should be zero
        assert (s_out[:, -3:, :] == 0).all()
        # Padded rows/cols in z should be zero
        assert (z_out[:, -3:, :, :] == 0).all()
        assert (z_out[:, :, -3:, :] == 0).all()


# ---------------------------------------------------------------------------
# AdaptorModule tests
# ---------------------------------------------------------------------------


class TestAdaptorModuleTrunk:
    """Tests for AdaptorModule with source_mode='trunk'."""

    def test_output_shapes_linear_only(self, trunk_inputs, default_dims):
        """Linear-only adaptor (n_attn_layers=0) should produce correct shapes."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=0,
            **default_dims,
        )
        s, z = adaptor(
            trunk_seqs=trunk_inputs["trunk_seqs"],
            trunk_pair=trunk_inputs["trunk_pair"],
            local_latents=trunk_inputs["local_latents"],
            ca_coords=trunk_inputs["ca_coords"],
        )

        b, n = trunk_inputs["trunk_seqs"].shape[:2]
        assert s.shape == (b, n, default_dims["target_s_dim"])
        assert z.shape == (b, n, n, default_dims["target_z_dim"])

    def test_output_shapes_with_attention(self, trunk_inputs, default_dims):
        """Attention-augmented adaptor should produce correct shapes."""
        for n_layers in [1, 2]:
            adaptor = AdaptorModule(
                source_mode="trunk",
                n_attn_layers=n_layers,
                **default_dims,
            )
            s, z = adaptor(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=trunk_inputs["ca_coords"],
                mask=trunk_inputs["mask"],
            )

            b, n = trunk_inputs["trunk_seqs"].shape[:2]
            assert s.shape == (b, n, default_dims["target_s_dim"]), (
                f"Failed for n_attn_layers={n_layers}"
            )
            assert z.shape == (b, n, n, default_dims["target_z_dim"]), (
                f"Failed for n_attn_layers={n_layers}"
            )

    def test_no_decoder_fusion_attribute(self, default_dims):
        """Trunk mode should NOT have decoder_fusion."""
        adaptor = AdaptorModule(source_mode="trunk", **default_dims)
        assert not hasattr(adaptor, "decoder_fusion")

    def test_default_mask_when_none(self, trunk_inputs, default_dims):
        """When mask is None, attention layers should create an all-ones mask."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=1,
            **default_dims,
        )
        # Should not raise
        s, z = adaptor(
            trunk_seqs=trunk_inputs["trunk_seqs"],
            trunk_pair=trunk_inputs["trunk_pair"],
            local_latents=trunk_inputs["local_latents"],
            ca_coords=trunk_inputs["ca_coords"],
            mask=None,
        )
        b, n = trunk_inputs["trunk_seqs"].shape[:2]
        assert s.shape == (b, n, default_dims["target_s_dim"])

    def test_linear_only_no_attn_blocks(self, default_dims):
        """n_attn_layers=0 should not create attn_blocks."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=0,
            **default_dims,
        )
        assert not hasattr(adaptor, "attn_blocks")

    def test_attention_blocks_created(self, default_dims):
        """n_attn_layers>0 should create the correct number of blocks."""
        for n_layers in [1, 2]:
            adaptor = AdaptorModule(
                source_mode="trunk",
                n_attn_layers=n_layers,
                **default_dims,
            )
            assert hasattr(adaptor, "attn_blocks")
            assert len(adaptor.attn_blocks) == n_layers

    def test_zero_init_attention_matches_linear(self, trunk_inputs, default_dims):
        """At init, attention-augmented adaptor should match linear-only adaptor.

        Because attention blocks are zero-initialised, their residual
        connections make them near-identity, so the overall output should
        be nearly identical to the linear-only case.
        """
        torch.manual_seed(42)
        adaptor_linear = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=0,
            **default_dims,
        )

        torch.manual_seed(42)
        adaptor_attn = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=2,
            **default_dims,
        )
        # Copy projection weights from linear to attn adaptor
        adaptor_attn.single_proj.load_state_dict(
            adaptor_linear.single_proj.state_dict()
        )
        adaptor_attn.pair_proj.load_state_dict(
            adaptor_linear.pair_proj.state_dict()
        )

        with torch.no_grad():
            s_lin, z_lin = adaptor_linear(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=trunk_inputs["ca_coords"],
                mask=trunk_inputs["mask"],
            )
            s_attn, z_attn = adaptor_attn(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=trunk_inputs["ca_coords"],
                mask=trunk_inputs["mask"],
            )

        torch.testing.assert_close(s_lin, s_attn, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_lin, z_attn, atol=1e-5, rtol=1e-5)

    def test_gradients_flow(self, trunk_inputs, default_dims):
        """Gradients should flow through all adaptor parameters."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=2,
            **default_dims,
        )
        s, z = adaptor(
            trunk_seqs=trunk_inputs["trunk_seqs"],
            trunk_pair=trunk_inputs["trunk_pair"],
            local_latents=trunk_inputs["local_latents"],
            ca_coords=trunk_inputs["ca_coords"],
            mask=trunk_inputs["mask"],
        )
        loss = s.sum() + z.sum()
        loss.backward()

        for name, param in adaptor.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestAdaptorModuleHybrid:
    """Tests for AdaptorModule with source_mode='hybrid'."""

    def test_output_shapes(self, trunk_inputs, default_dims):
        """Hybrid mode should produce correct output shapes."""
        b, n = trunk_inputs["trunk_seqs"].shape[:2]
        decoder_seqs = torch.randn(b, n, default_dims["trunk_dim"])

        adaptor = AdaptorModule(
            source_mode="hybrid",
            n_attn_layers=1,
            **default_dims,
        )
        s, z = adaptor(
            trunk_seqs=trunk_inputs["trunk_seqs"],
            trunk_pair=trunk_inputs["trunk_pair"],
            local_latents=trunk_inputs["local_latents"],
            ca_coords=trunk_inputs["ca_coords"],
            decoder_seqs=decoder_seqs,
            mask=trunk_inputs["mask"],
        )

        assert s.shape == (b, n, default_dims["target_s_dim"])
        assert z.shape == (b, n, n, default_dims["target_z_dim"])

    def test_has_decoder_fusion(self, default_dims):
        """Hybrid mode should have decoder_fusion."""
        adaptor = AdaptorModule(source_mode="hybrid", **default_dims)
        assert hasattr(adaptor, "decoder_fusion")

    def test_zero_init_hybrid_matches_trunk(self, trunk_inputs, default_dims):
        """At init, hybrid mode should match trunk mode.

        Because decoder_fusion is zero-initialised, the fused seqs equal
        trunk seqs, making hybrid == trunk at initialisation.
        """
        b, n = trunk_inputs["trunk_seqs"].shape[:2]
        decoder_seqs = torch.randn(b, n, default_dims["trunk_dim"])

        torch.manual_seed(42)
        adaptor_trunk = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=1,
            **default_dims,
        )

        torch.manual_seed(42)
        adaptor_hybrid = AdaptorModule(
            source_mode="hybrid",
            n_attn_layers=1,
            **default_dims,
        )
        # Copy projection + attn weights from trunk to hybrid
        trunk_state = adaptor_trunk.state_dict()
        adaptor_hybrid.load_state_dict(trunk_state, strict=False)

        with torch.no_grad():
            s_trunk, z_trunk = adaptor_trunk(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=trunk_inputs["ca_coords"],
                mask=trunk_inputs["mask"],
            )
            s_hybrid, z_hybrid = adaptor_hybrid(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=trunk_inputs["ca_coords"],
                decoder_seqs=decoder_seqs,
                mask=trunk_inputs["mask"],
            )

        torch.testing.assert_close(s_trunk, s_hybrid, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_trunk, z_hybrid, atol=1e-5, rtol=1e-5)

    def test_hybrid_without_decoder_seqs_falls_back(
        self, trunk_inputs, default_dims
    ):
        """If decoder_seqs is None in hybrid mode, should fall back to trunk."""
        adaptor = AdaptorModule(
            source_mode="hybrid",
            n_attn_layers=0,
            **default_dims,
        )
        # Should not raise
        s, z = adaptor(
            trunk_seqs=trunk_inputs["trunk_seqs"],
            trunk_pair=trunk_inputs["trunk_pair"],
            local_latents=trunk_inputs["local_latents"],
            ca_coords=trunk_inputs["ca_coords"],
            decoder_seqs=None,
        )
        b, n = trunk_inputs["trunk_seqs"].shape[:2]
        assert s.shape == (b, n, default_dims["target_s_dim"])

    def test_ca_distogram_is_added_to_pair_repr(self, trunk_inputs, default_dims):
        """Different C-alpha coordinates should change pair output z."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=0,
            **default_dims,
        )

        b, n = trunk_inputs["ca_coords"].shape[:2]
        ca_coords_a = torch.zeros_like(trunk_inputs["ca_coords"])
        ca_coords_b = torch.zeros_like(trunk_inputs["ca_coords"])
        ca_coords_b[:, :, 0] = torch.arange(
            n, device=ca_coords_b.device, dtype=ca_coords_b.dtype
        )[None, :].expand(b, -1)

        with torch.no_grad():
            _, z_a = adaptor(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=ca_coords_a,
            )
            _, z_b = adaptor(
                trunk_seqs=trunk_inputs["trunk_seqs"],
                trunk_pair=trunk_inputs["trunk_pair"],
                local_latents=trunk_inputs["local_latents"],
                ca_coords=ca_coords_b,
            )

        assert not torch.allclose(z_a, z_b)


class TestAdaptorModuleParameterCounts:
    """Tests for adaptor parameter counts."""

    def test_linear_only_param_count(self, default_dims):
        """Linear-only adaptor should have ~331K params (from architecture doc)."""
        adaptor = AdaptorModule(
            source_mode="trunk",
            n_attn_layers=0,
            **default_dims,
        )
        total = sum(p.numel() for p in adaptor.parameters())

        # single_proj: LayerNorm(776) + Linear(776, 384, bias=False)
        # = 776*2 + 776*384 = 1552 + 297984 = 299536
        # pair_proj: LayerNorm(256) + Linear(256, 128, bias=False)
        # = 256*2 + 256*128 = 512 + 32768 = 33280
        # Total ≈ 332816
        expected_single = 776 * 2 + 776 * 384  # LayerNorm(weight+bias) + Linear
        expected_pair = 256 * 2 + 256 * 128
        expected_total = expected_single + expected_pair

        assert total == expected_total, (
            f"Expected {expected_total} params, got {total}"
        )

    def test_attention_adds_parameters(self, default_dims):
        """Adding attention layers should increase parameter count."""
        adaptor_0 = AdaptorModule(
            source_mode="trunk", n_attn_layers=0, **default_dims
        )
        adaptor_1 = AdaptorModule(
            source_mode="trunk", n_attn_layers=1, **default_dims
        )
        adaptor_2 = AdaptorModule(
            source_mode="trunk", n_attn_layers=2, **default_dims
        )

        p0 = sum(p.numel() for p in adaptor_0.parameters())
        p1 = sum(p.numel() for p in adaptor_1.parameters())
        p2 = sum(p.numel() for p in adaptor_2.parameters())

        assert p1 > p0
        assert p2 > p1
        # 2-layer should add exactly twice the params of 1-layer (over base)
        assert p2 - p0 == 2 * (p1 - p0)

    def test_hybrid_adds_decoder_fusion_params(self, default_dims):
        """Hybrid mode should have more params than trunk mode (decoder_fusion)."""
        adaptor_trunk = AdaptorModule(
            source_mode="trunk", n_attn_layers=1, **default_dims
        )
        adaptor_hybrid = AdaptorModule(
            source_mode="hybrid", n_attn_layers=1, **default_dims
        )

        p_trunk = sum(p.numel() for p in adaptor_trunk.parameters())
        p_hybrid = sum(p.numel() for p in adaptor_hybrid.parameters())

        # decoder_fusion: LayerNorm(768) + Linear(768, 768, bias=False)
        # = 768*2 + 768*768 = 1536 + 589824 = 591360
        expected_extra = 768 * 2 + 768 * 768
        assert p_hybrid - p_trunk == expected_extra
