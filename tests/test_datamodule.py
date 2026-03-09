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


class TestBoltzPassSkipsExisting:
    """Test that the Boltz pass skips graphs that already have pLDDT."""

    def test_skip_if_plddt_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()

            # Save a graph WITH plddt already
            graph = _make_fake_graph("test_pdb", has_plddt=True)
            torch.save(graph, processed / "test_pdb.pt")

            # Create a minimal CSV
            import pandas as pd
            df = pd.DataFrame({"pdb": ["test_pdb"], "id": ["test_pdb"]})
            df.to_csv(Path(tmpdir) / "test.csv", index=False)

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={},
            )
            # _run_boltz_pass should detect existing plddt and skip
            with patch.object(dm, "_run_boltz_for_structure") as mock_boltz:
                dm._run_boltz_pass(["test_pdb.pt"])
                mock_boltz.assert_not_called()


class TestBoltzPassProcessesNew:
    """Test that the Boltz pass runs on graphs without pLDDT."""

    def test_processes_new_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            raw = Path(tmpdir) / "raw"
            raw.mkdir()

            # Save a graph WITHOUT plddt
            graph = _make_fake_graph("test_pdb", n_residues=5, has_plddt=False)
            torch.save(graph, processed / "test_pdb.pt")

            # Create a dummy CIF file
            (raw / "test_pdb.cif").write_text("dummy")

            dm = QualityGraftDataModule(
                data_dir=tmpdir,
                boltz_config={"model": "boltz1", "devices": 1, "accelerator": "cpu"},
            )

            fake_plddt = np.random.rand(5).astype(np.float32)
            with patch.object(dm, "_run_boltz_for_structure", return_value=fake_plddt):
                dm._run_boltz_pass(["test_pdb.pt"])

            # Reload and check
            updated = torch.load(processed / "test_pdb.pt", weights_only=False)
            assert hasattr(updated, "plddt")
            assert hasattr(updated, "plddt_bin")
            assert updated.plddt.shape[0] == 5
            assert updated.plddt_bin.shape[0] == 5
            assert updated.plddt_bin.dtype == torch.long
