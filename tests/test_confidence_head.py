"""Unit tests for BoltzConfidenceHead."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import quality_graft.models.confidence_head as confidence_head_module


class _DummyConfidenceModule(torch.nn.Module):
    """Minimal stand-in for Boltz ConfidenceModule for loading tests."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        return SimpleNamespace(missing_keys=["missing.weight"], unexpected_keys=[])


def test_strict_loading_raises_on_missing_keys(monkeypatch, tmp_path):
    """Strict loading must fail when checkpoint keys are incomplete."""
    monkeypatch.setattr(confidence_head_module, "ConfidenceModule", _DummyConfidenceModule)

    def _fake_torch_load(*args, **kwargs):
        return {"state_dict": {"confidence_module.some.weight": torch.zeros(1)}}

    monkeypatch.setattr(torch, "load", _fake_torch_load)

    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"x")

    with pytest.raises(RuntimeError, match="Confidence weight loading mismatch"):
        confidence_head_module.BoltzConfidenceHead(
            token_s=384,
            token_z=128,
            pairformer_args={},
            confidence_model_args={"confidence_args": {}},
            full_embedder_args={},
            ckpt_path=str(ckpt_path),
            ckpt_prefix="confidence_module.",
            device="cpu",
            strict_loading=True,
        )
