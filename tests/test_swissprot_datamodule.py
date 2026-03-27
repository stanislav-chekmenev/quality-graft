"""Tests for SwissProtDataModule."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from quality_graft.data.swissprot_datamodule import SwissProtDataModule
from quality_graft.data.swissprot_selector import SwissProtDataSelector
from quality_graft.data.datamodule import _load_plddt_status


def _make_metadata_tsv(path, entries):
    """Write a fake UniProt metadata TSV."""
    df = pd.DataFrame(entries, columns=["accession", "length"])
    df.to_csv(path, sep="\t", index=False)


def _make_pdb_file(directory, accession, n_residues=10, version=4):
    """Create a minimal but parseable PDB file.

    Creates a single-chain protein with n_residues GLY residues,
    B-factor column set to 85.0 (simulating pLDDT=0.85).
    """
    fname = f"AF-{accession}-F1-model_v{version}.pdb"
    lines = []
    for i in range(n_residues):
        # Standard ATOM record format (PDB spec)
        # Columns: ATOM, serial, name, resName, chainID, resSeq, x, y, z, occupancy, bfactor
        x, y, z = float(i), 0.0, 0.0
        # Write all 37 atom types that graphein expects? No, just CA is enough for protein_to_pyg
        line = (
            f"ATOM  {i+1:5d}  CA  GLY A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00 85.00           C  "
        )
        lines.append(line)
    lines.append("END")
    (directory / fname).write_text("\n".join(lines) + "\n")
    return fname


def _setup_swissprot_dirs(tmpdir, accessions, n_residues=10):
    """Create directory structure with metadata TSV and PDB files."""
    data_dir = Path(tmpdir) / "data"
    data_dir.mkdir()
    source_dir = Path(tmpdir) / "source"
    source_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    tsv_path = Path(tmpdir) / "metadata.tsv"
    entries = [(acc, n_residues * 3) for acc in accessions]  # length > n_residues
    _make_metadata_tsv(tsv_path, entries)

    # Create PDB files in source AND raw (simulating copy_swissprot already ran)
    for acc in accessions:
        _make_pdb_file(source_dir, acc, n_residues=n_residues)
        _make_pdb_file(raw_dir, acc, n_residues=n_residues)

    return data_dir, source_dir, tsv_path


class TestSwissProtDataModuleInheritance:
    def test_inherits_quality_graft_data_module(self):
        from quality_graft.data.datamodule import QualityGraftDataModule
        assert issubclass(SwissProtDataModule, QualityGraftDataModule)


class TestGetFileIdentifier:
    def test_returns_swissprot_specific_string(self):
        """_get_file_identifier returns a SwissProt-specific string, not the PDB one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [("P12345", 100)])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
                fraction=0.5,
                min_length=30,
                max_length=512,
            )

            from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSplitter
            splitter = PDBDataSplitter(data_dir=str(data_dir))

            dm = SwissProtDataModule(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                dataselector=selector,
                datasplitter=splitter,
                format="pdb",
                boltz_config={},
            )

            result = dm._get_file_identifier(selector)
            assert "swissprot" in result
            assert "f0.5" in result
            assert "minl30" in result
            assert "maxl512" in result
            # Should NOT contain PDB-specific fields
            assert "molecule_type" not in result
            assert "oligomeric" not in result


class TestPrepareDataNoSuperCall:
    def test_does_not_call_super_prepare_data(self):
        """prepare_data must NOT call super().prepare_data() (would trigger Boltz)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, source_dir, tsv_path = _setup_swissprot_dirs(tmpdir, ["P12345"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSplitter
            splitter = PDBDataSplitter(data_dir=str(data_dir))

            dm = SwissProtDataModule(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                dataselector=selector,
                datasplitter=splitter,
                format="pdb",
                boltz_config={},
            )

            with patch("quality_graft.data.datamodule.QualityGraftDataModule.prepare_data") as mock_super:
                dm.prepare_data()
                mock_super.assert_not_called()


class TestPrepareDataWritesPlddtStatus:
    def test_plddt_status_csv_created(self):
        """prepare_data writes plddt_status.csv with all processed structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, source_dir, tsv_path = _setup_swissprot_dirs(tmpdir, ["P12345", "Q67890"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSplitter
            splitter = PDBDataSplitter(data_dir=str(data_dir))

            dm = SwissProtDataModule(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                dataselector=selector,
                datasplitter=splitter,
                format="pdb",
                boltz_config={},
            )

            dm.prepare_data()

            status = _load_plddt_status(data_dir / "processed" / "plddt_status.csv")
            # At least one structure should be marked as having pLDDT
            assert len(status) > 0
            assert all(v is True for v in status.values())


class TestProcessedGraphHasPlddtFields:
    def test_pt_files_have_plddt_fields(self):
        """Processed .pt files must have plddt, plddt_bin, plddt_logits, database fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir, source_dir, tsv_path = _setup_swissprot_dirs(tmpdir, ["P12345"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSplitter
            splitter = PDBDataSplitter(data_dir=str(data_dir))

            dm = SwissProtDataModule(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                dataselector=selector,
                datasplitter=splitter,
                format="pdb",
                boltz_config={},
            )

            dm.prepare_data()

            pt_files = list((data_dir / "processed").glob("*.pt"))
            assert len(pt_files) > 0

            graph = torch.load(pt_files[0], weights_only=False)
            assert hasattr(graph, "plddt")
            assert hasattr(graph, "plddt_bin")
            # hard targets only — PyG Data doesn't persist None attrs through save/load
            assert not hasattr(graph, "plddt_logits") or graph.plddt_logits is None
            assert graph.database == "swissprot"
            # B-factor is 85.0 → pLDDT = 0.85
            assert torch.all(graph.plddt >= 0.0)
            assert torch.all(graph.plddt <= 1.0)
            assert graph.plddt_bin.dtype == torch.long


class TestPrepareDataEmptyDataset:
    def test_raises_on_empty_dataset(self):
        """prepare_data raises ValueError if selector returns zero results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [])  # empty

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSplitter
            splitter = PDBDataSplitter(data_dir=str(data_dir))

            dm = SwissProtDataModule(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                dataselector=selector,
                datasplitter=splitter,
                format="pdb",
                boltz_config={},
            )

            with pytest.raises(ValueError):
                dm.prepare_data()
