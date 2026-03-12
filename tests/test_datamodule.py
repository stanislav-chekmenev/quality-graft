"""Tests for QualityGraftDataModule.

Unit tests mock the Boltz runner to avoid GPU/network dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from quality_graft.data.datamodule import QualityGraftDataModule
from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult


def _make_fake_graph(pdb_id: str, n_residues: int = 10, has_plddt: bool = False) -> Data:
    """Create a minimal PyG Data object mimicking PDBDataset output."""
    graph = Data()
    graph.id = pdb_id
    graph.coords = torch.randn(n_residues, 37, 3)
    graph.coord_mask = torch.ones(n_residues, 37, dtype=torch.bool)
    graph.residue_type = torch.randint(0, 20, (n_residues,))
    graph.seq_pos = torch.arange(n_residues).unsqueeze(-1)
    graph.database = "pdb"
    if has_plddt:
        graph.plddt = torch.rand(n_residues)
        graph.plddt_bin = torch.randint(0, 50, (n_residues,))
    return graph


def _make_batch_result(structure_plddt_map: dict[str, np.ndarray | None]) -> BoltzBatchResult:
    """Build a BoltzBatchResult from a {structure_id: plddt_array} dict."""
    results = {}
    for sid, plddt in structure_plddt_map.items():
        if plddt is not None:
            results[sid] = BoltzResult(
                structure_id=sid,
                plddt=plddt,
                confidence_json=None,
                success=True,
                error_msg=None,
            )
    return BoltzBatchResult(
        results=results,
        n_submitted=len(structure_plddt_map),
        returncode=0,
        error_msg=None,
    )


class TestBoltzPassSkipsExisting:
    """Test that the Boltz pass skips graphs that already have pLDDT."""

    def test_skip_if_plddt_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            graph = _make_fake_graph("test_pdb", has_plddt=True)
            torch.save(graph, processed / "test_pdb.pt")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )

            with patch("quality_graft.data.boltz_runner.run_boltz_predict_dir") as mock_batch:
                dm._run_boltz_pass(["test_pdb.pt"])
                mock_batch.assert_not_called()


class TestBoltzPassProcessesNew:
    """Test that the Boltz pass runs on graphs without pLDDT."""

    def test_processes_new_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            graph = _make_fake_graph("test_pdb", n_residues=5, has_plddt=False)
            torch.save(graph, processed / "test_pdb.pt")

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            fake_result = _make_batch_result({"test_pdb": fake_plddt})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.boltz_runner.run_boltz_predict_dir", return_value=fake_result):
                dm._run_boltz_pass(["test_pdb.pt"])

            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt")
            assert hasattr(updated, "plddt_bin")
            assert updated.plddt.shape[0] == 5
            assert updated.plddt_bin.shape[0] == 5
            assert updated.plddt_bin.dtype == torch.long


class TestBoltzPassPartialFailure:
    """Test that partial Boltz failures are handled gracefully."""

    def test_missing_result_counted_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            for sid in ["good_A", "bad_B"]:
                graph = _make_fake_graph(sid, n_residues=5, has_plddt=False)
                torch.save(graph, processed / f"{sid}.pt")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            fake_result = _make_batch_result({"good_A": fake_plddt, "bad_B": None})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.boltz_runner.run_boltz_predict_dir", return_value=fake_result):
                dm._run_boltz_pass(["good_A.pt", "bad_B.pt"])

            good = torch.load(processed / "good_A.pt", weights_only=False)
            bad = torch.load(processed / "bad_B.pt", weights_only=False)

            assert hasattr(good, "plddt")
            assert not hasattr(bad, "plddt") or bad.plddt is None


class TestPrepareYamlOutputDir:
    """Test that _prepare_boltz_yaml writes to custom output_dir."""

    def test_writes_to_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)
            custom_dir = Path(tmpdir) / "boltz_work" / "inputs" / "chunk_000"
            custom_dir.mkdir(parents=True)

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"use_msa_server": False},
            )

            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[{"sequence": "ACGT"}]), \
                 patch("quality_graft.data.datamodule.chains_to_boltz_yaml", return_value="dummy_yaml"):
                result = dm._prepare_boltz_yaml("test_A", "test", output_dir=custom_dir)

            assert result is not None
            assert result.parent == custom_dir
            assert (custom_dir / "test_A.yaml").exists()

    def test_defaults_to_boltz_inputs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"use_msa_server": False},
            )

            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[{"sequence": "ACGT"}]), \
                 patch("quality_graft.data.datamodule.chains_to_boltz_yaml", return_value="dummy_yaml"):
                result = dm._prepare_boltz_yaml("test_A", "test")

            assert result is not None
            assert result.parent == boltz_inputs
