"""Tests for W&B logging integration in QualityGraftDataModule."""

import time
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from pathlib import Path


def _make_graph(n_residues=50, has_plddt=False):
    """Create a minimal PyG Data object."""
    graph = Data(coords=torch.randn(n_residues, 3))
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.zeros(n_residues, dtype=torch.long)
    return graph


@pytest.fixture
def mock_datamodule(tmp_path):
    """Create a minimal mock of QualityGraftDataModule with the methods we need."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    # Save a .pt file without pLDDT so it gets processed
    graph = _make_graph(n_residues=50, has_plddt=False)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    torch.save(graph, processed_dir / "test_structure.pt")

    # We can't easily instantiate the full datamodule (needs PDB dependencies),
    # so we test the _run_boltz_pass logic via patching
    return processed_dir


@patch("quality_graft.data.datamodule.log_protein_metrics")
def test_run_boltz_pass_calls_log_protein_metrics(mock_log, mock_datamodule):
    """Verify that _run_boltz_pass calls log_protein_metrics for successful structures."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    processed_dir = mock_datamodule

    # Create a mock datamodule instance with required attributes
    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = processed_dir
    dm.num_plddt_bins = 50

    # Mock _run_boltz_for_structure to return a valid pLDDT array
    fake_plddt = np.random.rand(50).astype(np.float32)
    dm._run_boltz_for_structure = MagicMock(return_value=fake_plddt)

    # Call the real method
    QualityGraftDataModule._run_boltz_pass(dm, ["test_structure.pt"])

    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args
    # Positional args: structure_id, plddt, n_residues, elapsed_s, n_processed, n_failed, n_skipped
    args = call_kwargs[1] if call_kwargs[1] else call_kwargs[0]
    if isinstance(args, tuple):
        assert args[0] == "test_structure"  # structure_id
        assert args[2] == 50  # n_residues


@patch("quality_graft.data.datamodule.log_protein_metrics")
def test_run_boltz_pass_skips_logging_for_already_labeled(mock_log, tmp_path):
    """Verify that already-labeled structures are skipped (no logging)."""
    from quality_graft.data.datamodule import QualityGraftDataModule

    # Save a .pt file WITH pLDDT (already labeled)
    graph = _make_graph(n_residues=50, has_plddt=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    torch.save(graph, processed_dir / "labeled_structure.pt")

    dm = MagicMock(spec=QualityGraftDataModule)
    dm.processed_dir = processed_dir
    dm.num_plddt_bins = 50

    QualityGraftDataModule._run_boltz_pass(dm, ["labeled_structure.pt"])

    mock_log.assert_not_called()
