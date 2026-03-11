"""Tests verifying datamodule does NOT call per-protein W&B logging."""

import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import pytest

from quality_graft.data.datamodule import QualityGraftDataModule
from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult


def _make_graph(n_residues=50, has_plddt=False):
    """Create a minimal PyG Data object."""
    graph = Data(coords=torch.randn(n_residues, 3))
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.zeros(n_residues, dtype=torch.long)
    return graph


def test_run_boltz_pass_saves_plddt(tmp_path):
    """Verify pLDDT is saved correctly without W&B dependency."""
    processed = tmp_path / "processed"
    processed.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    boltz_inputs = tmp_path / "boltz_work" / "inputs"
    boltz_inputs.mkdir(parents=True)

    graph = _make_graph(n_residues=50, has_plddt=False)
    torch.save(graph, processed / "test_structure.pt")

    dm = QualityGraftDataModule(
        data_dir=str(tmp_path),
        boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
    )

    fake_plddt = np.random.rand(50).astype(np.float32)
    fake_result = BoltzBatchResult(
        results={
            "test_structure": BoltzResult(
                structure_id="test_structure",
                plddt=fake_plddt,
                confidence_json=None,
                success=True,
                error_msg=None,
            )
        },
        n_submitted=1,
        returncode=0,
        error_msg=None,
    )

    with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
         patch("quality_graft.data.boltz_runner.run_boltz_predict_dir", return_value=fake_result):
        dm._run_boltz_pass(["test_structure.pt"])

    updated = torch.load(processed / "test_structure.pt", weights_only=False)
    assert hasattr(updated, "plddt")
    assert hasattr(updated, "plddt_bin")
    assert updated.plddt.shape[0] == 50


def test_run_boltz_pass_no_wandb_import():
    """Verify datamodule does not import log_protein_metrics (removed)."""
    import quality_graft.data.datamodule as dm_mod
    assert not hasattr(dm_mod, "log_protein_metrics")
