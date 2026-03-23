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


def _make_batch_result(
    structure_plddt_map: dict[str, np.ndarray | None],
    include_logits: bool = False,
) -> BoltzBatchResult:
    """Build a BoltzBatchResult from a {structure_id: plddt_array} dict."""
    results = {}
    for sid, plddt in structure_plddt_map.items():
        if plddt is not None:
            n = plddt.shape[0]
            results[sid] = BoltzResult(
                structure_id=sid,
                plddt=plddt,
                plddt_logits=np.random.randn(n, 50).astype(np.float32) if include_logits else None,
                pde_logits=np.random.randn(n, n, 64).astype(np.float32) if include_logits else None,
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

            with patch("quality_graft.data.datamodule.run_boltz_predict_dir") as mock_batch:
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

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
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

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                result_map = {}
                for sid in structure_ids:
                    if sid == "good_A":
                        result_map[sid] = fake_plddt.copy()
                    else:
                        result_map[sid] = None
                return _make_batch_result(result_map)

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
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

            mock_chain = MagicMock(chain_id="A", sequence="ACGT", n_residues=4)
            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[mock_chain]), \
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

            mock_chain = MagicMock(chain_id="A", sequence="ACGT", n_residues=4)
            with patch("quality_graft.data.datamodule.parse_cif_chains", return_value=[mock_chain]), \
                 patch("quality_graft.data.datamodule.chains_to_boltz_yaml", return_value="dummy_yaml"):
                result = dm._prepare_boltz_yaml("test_A", "test")

            assert result is not None
            assert result.parent == boltz_inputs


class TestChunkedBoltzPass:
    """Test sequential chunked Boltz execution."""

    def _setup_structures(self, tmpdir, structure_ids, n_residues=5):
        """Helper to create .pt files and raw CIFs for testing."""
        processed = Path(tmpdir) / "processed"
        processed.mkdir(exist_ok=True)
        raw = Path(tmpdir) / "raw"
        raw.mkdir(exist_ok=True)
        boltz_work = Path(tmpdir) / "boltz_work"
        boltz_work.mkdir(exist_ok=True)
        (boltz_work / "inputs").mkdir(exist_ok=True)

        for sid in structure_ids:
            graph = _make_fake_graph(sid, n_residues=n_residues, has_plddt=False)
            torch.save(graph, processed / f"{sid}.pt")
            pdb_code = sid.split("_")[0]
            (raw / f"{pdb_code}.cif").write_text("dummy")

        return processed

    def test_structures_split_into_chunks(self):
        """Verify run_boltz_predict_dir is called once per chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(25)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 10,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir) as mock_batch:
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # 25 structures / chunk_size 10 = 3 chunks
            assert mock_batch.call_count == 3

            # Verify all structures got pLDDT
            processed = Path(tmpdir) / "processed"
            for sid in sids:
                graph = torch.load(processed / f"{sid}.pt", weights_only=False)
                assert hasattr(graph, "plddt"), f"{sid} missing plddt"
                assert hasattr(graph, "plddt_bin"), f"{sid} missing plddt_bin"

    def test_chunk_failure_doesnt_block_others(self):
        """One chunk failing should not prevent other chunks from saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(20)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 10,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                # Fail chunk_000 deterministically by directory name
                if input_dir.name == "chunk_000":
                    return BoltzBatchResult(
                        results={}, n_submitted=len(structure_ids),
                        returncode=1, error_msg="Boltz OOM: GPU memory exhaustion",
                    )
                # All other chunks succeed
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # At least some structures should have pLDDT (from the successful chunk)
            processed = Path(tmpdir) / "processed"
            labeled = 0
            for sid in sids:
                graph = torch.load(processed / f"{sid}.pt", weights_only=False)
                if hasattr(graph, "plddt") and graph.plddt is not None:
                    labeled += 1
            assert labeled == 10  # One chunk of 10 succeeded

    def test_each_chunk_gets_own_directories(self):
        """Verify each chunk gets unique input and output dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sids = [f"pdb{i}_A" for i in range(5)]
            self._setup_structures(tmpdir, sids)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={
                    "chunk_size": 3,
                },
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            seen_input_dirs = []
            seen_out_dirs = []

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                seen_input_dirs.append(input_dir)
                seen_out_dirs.append(out_dir)
                return _make_batch_result({sid: fake_plddt.copy() for sid in structure_ids})

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass([f"{sid}.pt" for sid in sids])

            # 5 structures / chunk_size 3 = 2 chunks
            assert len(seen_input_dirs) == 2
            assert len(set(seen_input_dirs)) == 2  # All unique
            assert len(set(seen_out_dirs)) == 2  # All unique
            # Verify chunk naming
            for d in seen_input_dirs:
                assert d.name.startswith("chunk_")


class TestBoltzPassSavesLogits:
    """Test that logits from Boltz are saved into .pt files."""

    def test_plddt_and_pde_logits_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            n_residues = 5
            graph = _make_fake_graph("test_pdb", n_residues=n_residues, has_plddt=False)
            torch.save(graph, processed / "test_pdb.pt")

            (raw / "test.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(n_residues).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result(
                    {sid: fake_plddt.copy() for sid in structure_ids},
                    include_logits=True,
                )

            with patch.object(dm, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir):
                dm._run_boltz_pass(["test_pdb.pt"])

            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt_logits")
            assert updated.plddt_logits.shape == (n_residues, 50)
            assert hasattr(updated, "pde_logits")
            assert updated.pde_logits.shape == (n_residues, n_residues, 64)


class TestReprocessBoltzMode:
    """Test reprocess_boltz=True forces re-processing of all structures."""

    def test_reprocess_overrides_existing_plddt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            boltz_inputs = Path(tmpdir) / "boltz_work" / "inputs"
            boltz_inputs.mkdir(parents=True)

            n_residues = 5
            graph = _make_fake_graph("test_pdb", n_residues=n_residues, has_plddt=True)
            torch.save(graph, processed / "test_pdb.pt")

            (raw / "test.cif").write_text("dummy")

            # Mark as already having pLDDT in status file
            from quality_graft.data.datamodule import _save_plddt_status
            _save_plddt_status(processed / "plddt_status.csv", {"test_pdb": True})

            # Without reprocess_boltz, should skip
            dm_skip = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
                reprocess_boltz=False,
            )
            with patch("quality_graft.data.datamodule.run_boltz_predict_dir") as mock_batch:
                dm_skip._run_boltz_pass(["test_pdb.pt"])
                mock_batch.assert_not_called()

            # With reprocess_boltz, should re-process
            dm_reprocess = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
                reprocess_boltz=True,
            )
            fake_plddt = np.random.rand(n_residues).astype(np.float32)

            def mock_predict_dir(input_dir, out_dir, structure_ids, **kwargs):
                return _make_batch_result(
                    {sid: fake_plddt.copy() for sid in structure_ids},
                    include_logits=True,
                )

            with patch.object(dm_reprocess, "_prepare_boltz_yaml", return_value=Path("dummy.yaml")), \
                 patch("quality_graft.data.datamodule.run_boltz_predict_dir", side_effect=mock_predict_dir) as mock_batch:
                dm_reprocess._run_boltz_pass(["test_pdb.pt"])
                mock_batch.assert_called_once()

            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt_logits")
            assert updated.plddt_logits.shape == (n_residues, 50)
            assert hasattr(updated, "pde_logits")
            assert updated.pde_logits.shape == (n_residues, n_residues, 64)
