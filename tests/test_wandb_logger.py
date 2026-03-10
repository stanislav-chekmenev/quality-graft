"""Tests for wandb_logger dataset stats collection."""

import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from torch_geometric.data import Data

from quality_graft.data.wandb_logger import collect_dataset_stats


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Create a temp directory with fake .pt files containing pLDDT labels."""
    for i in range(3):
        n_residues = 50 + i * 10
        plddt = torch.rand(n_residues)
        graph = Data(
            coords=torch.randn(n_residues, 3),
            plddt=plddt,
            plddt_bin=torch.zeros(n_residues, dtype=torch.long),
        )
        torch.save(graph, tmp_path / f"structure_{i}.pt")

    # One file without pLDDT (should be skipped)
    graph_no_plddt = Data(coords=torch.randn(20, 3))
    torch.save(graph_no_plddt, tmp_path / "no_plddt.pt")

    return tmp_path


def test_collect_dataset_stats_returns_all_labeled(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    assert len(stats) == 3  # only the 3 with pLDDT


def test_collect_dataset_stats_metric_keys(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    expected_keys = [
        "protein/structure_id",
        "protein/length",
        "protein/mean_plddt",
        "protein/median_plddt",
        "_plddt_array",
    ]
    for key in expected_keys:
        assert key in stats[0], f"Missing key: {key}"


def test_collect_dataset_stats_empty_dir(tmp_path):
    stats = collect_dataset_stats(tmp_path)
    assert stats == []


def test_collect_dataset_stats_plddt_array_is_numpy(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    for s in stats:
        assert isinstance(s["_plddt_array"], np.ndarray)
