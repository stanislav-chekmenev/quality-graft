"""Tests for slim wandb_logger dataset stats collection and summary logging."""

import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from torch_geometric.data import Data

from quality_graft.data.wandb_logger import collect_dataset_stats, log_dataset_summary


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Create a temp directory with fake .pt files containing pLDDT labels."""
    plddt_values = [
        torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),  # mean=0.7
        torch.tensor([0.95, 0.92, 0.88, 0.85, 0.80]),  # mean=0.88
        torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7]),  # mean=0.5
    ]
    for i, plddt in enumerate(plddt_values):
        graph = Data(
            coords=torch.randn(len(plddt), 3),
            plddt=plddt,
            plddt_bin=torch.zeros(len(plddt), dtype=torch.long),
        )
        torch.save(graph, tmp_path / f"structure_{i}.pt")

    # One file without pLDDT (should be skipped)
    graph_no_plddt = Data(coords=torch.randn(20, 3))
    torch.save(graph_no_plddt, tmp_path / "no_plddt.pt")

    return tmp_path


def test_collect_returns_only_labeled(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    assert len(stats) == 3


def test_collect_returns_tuples_with_correct_fields(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    for s in stats:
        assert "structure_id" in s
        assert "mean_plddt" in s
        assert "n_residues" in s
        assert isinstance(s["mean_plddt"], float)
        assert isinstance(s["n_residues"], int)


def test_collect_empty_dir(tmp_path):
    stats = collect_dataset_stats(tmp_path)
    assert stats == []


def test_collect_mean_plddt_values(tmp_processed_dir):
    stats = collect_dataset_stats(tmp_processed_dir)
    means = sorted([s["mean_plddt"] for s in stats])
    assert abs(means[0] - 0.5) < 0.01
    assert abs(means[1] - 0.7) < 0.01
    assert abs(means[2] - 0.88) < 0.01


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_logs_correct_keys(mock_wandb):
    mock_wandb.run = MagicMock()
    mock_wandb.Histogram = MagicMock(return_value="histogram_obj")

    stats = [
        {"structure_id": "a", "mean_plddt": 0.7, "n_residues": 5},
        {"structure_id": "b", "mean_plddt": 0.88, "n_residues": 5},
        {"structure_id": "c", "mean_plddt": 0.5, "n_residues": 5},
    ]
    log_dataset_summary(stats)

    mock_wandb.log.assert_called_once()
    logged = mock_wandb.log.call_args[0][0]

    expected_keys = {
        "dataset/plddt_histogram",
        "dataset/mean_plddt",
        "dataset/std_plddt",
        "dataset/max_plddt",
        "dataset/min_plddt",
        "dataset/num_proteins",
    }
    assert set(logged.keys()) == expected_keys


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_scalar_values(mock_wandb):
    mock_wandb.run = MagicMock()
    mock_wandb.Histogram = MagicMock(return_value="histogram_obj")

    stats = [
        {"structure_id": "a", "mean_plddt": 0.6, "n_residues": 5},
        {"structure_id": "b", "mean_plddt": 0.8, "n_residues": 5},
    ]
    log_dataset_summary(stats)

    logged = mock_wandb.log.call_args[0][0]
    assert logged["dataset/num_proteins"] == 2
    assert abs(logged["dataset/mean_plddt"] - 0.7) < 1e-6
    assert abs(logged["dataset/max_plddt"] - 0.8) < 1e-6
    assert abs(logged["dataset/min_plddt"] - 0.6) < 1e-6


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_noop_when_no_wandb_run(mock_wandb):
    mock_wandb.run = None
    stats = [{"structure_id": "a", "mean_plddt": 0.7, "n_residues": 5}]
    log_dataset_summary(stats)
    mock_wandb.log.assert_not_called()


@patch("quality_graft.data.wandb_logger.wandb")
def test_log_summary_noop_when_empty_stats(mock_wandb):
    mock_wandb.run = MagicMock()
    log_dataset_summary([])
    mock_wandb.log.assert_not_called()
