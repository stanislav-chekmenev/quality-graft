"""Tests verifying datamodule does NOT call per-protein W&B logging."""

from unittest.mock import MagicMock

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import pytest

from quality_graft.data.datamodule import QualityGraftDataModule


def _make_graph(n_residues=50, has_plddt=False):
    """Create a minimal PyG Data object."""
    graph = Data(coords=torch.randn(n_residues, 3))
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.zeros(n_residues, dtype=torch.long)
    return graph


@pytest.fixture
def mock_processed_dir(tmp_path):
    """Create processed dir with one unlabeled .pt file."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    graph = _make_graph(n_residues=50, has_plddt=False)
    torch.save(graph, processed_dir / "test_structure.pt")
    return processed_dir


def test_run_boltz_pass_saves_plddt(mock_processed_dir):
    """Verify pLDDT is saved correctly without W&B dependency."""
    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = mock_processed_dir
    dm.num_plddt_bins = 50

    fake_plddt = np.random.rand(50).astype(np.float32)
    dm._run_boltz_for_structure = MagicMock(return_value=fake_plddt)

    QualityGraftDataModule._run_boltz_pass(dm, ["test_structure.pt"])

    updated = torch.load(mock_processed_dir / "test_structure.pt", weights_only=False)
    assert hasattr(updated, "plddt")
    assert hasattr(updated, "plddt_bin")
    assert updated.plddt.shape[0] == 50


def test_run_boltz_pass_no_wandb_import():
    """Verify datamodule does not import log_protein_metrics (removed)."""
    import quality_graft.data.datamodule as dm_mod
    assert not hasattr(dm_mod, "log_protein_metrics")
