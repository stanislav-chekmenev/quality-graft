"""Tests for SwissProtDataSelector."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quality_graft.data.swissprot_selector import SwissProtDataSelector


def _make_metadata_tsv(path, entries):
    """Write a fake UniProt metadata TSV with accession and length columns."""
    df = pd.DataFrame(entries, columns=["accession", "length"])
    df.to_csv(path, sep="\t", index=False)


def _make_pdb_files(directory, accessions, version=4):
    """Create empty PDB files matching AlphaFold naming convention."""
    for acc in accessions:
        (directory / f"AF-{acc}-F1-model_v{version}.pdb").touch()


class TestSwissProtDataSelectorCreateDataset:
    def test_basic_filtering(self):
        """create_dataset returns DataFrame with expected columns and filters by file existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [
                ("P12345", 100),
                ("Q67890", 200),
                ("R11111", 50),   # too short (below min_length=60)
                ("S22222", 600),  # too long (above max_length=512)
            ])
            # Only create PDB files for P12345 and Q67890
            _make_pdb_files(source_dir, ["P12345", "Q67890"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
                min_length=60,
                max_length=512,
            )
            df = selector.create_dataset()

            assert "pdb" in df.columns
            assert "id" in df.columns
            assert "accession" in df.columns
            assert "length" in df.columns
            assert len(df) == 2
            assert set(df["accession"].tolist()) == {"P12345", "Q67890"}
            assert df.loc[df["accession"] == "P12345", "pdb"].iloc[0] == "AF-P12345-F1-model_v4"

    def test_exclude_ids(self):
        """Excluded IDs are removed from the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [
                ("P12345", 100),
                ("Q67890", 200),
            ])
            _make_pdb_files(source_dir, ["P12345", "Q67890"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
                exclude_ids=["P12345"],
            )
            df = selector.create_dataset()
            assert len(df) == 1
            assert df["accession"].iloc[0] == "Q67890"

    def test_exclude_ids_from_file(self):
        """Excluded IDs read from a file are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [
                ("P12345", 100),
                ("Q67890", 200),
            ])
            _make_pdb_files(source_dir, ["P12345", "Q67890"])

            exclude_file = Path(tmpdir) / "exclude.txt"
            exclude_file.write_text("Q67890\n")

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
                exclude_ids_from_file=str(exclude_file),
            )
            df = selector.create_dataset()
            assert len(df) == 1
            assert df["accession"].iloc[0] == "P12345"

    def test_fraction_sampling(self):
        """Fraction < 1.0 reduces the dataset size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            entries = [(f"P{i:05d}", 100) for i in range(100)]
            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, entries)
            _make_pdb_files(source_dir, [e[0] for e in entries])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
                fraction=0.5,
            )
            df = selector.create_dataset()
            assert 30 <= len(df) <= 70

    def test_no_chain_column(self):
        """DataFrame should NOT have a 'chain' column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [("P12345", 100)])
            _make_pdb_files(source_dir, ["P12345"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            df = selector.create_dataset()
            assert "chain" not in df.columns

    def test_no_sequence_column(self):
        """DataFrame should NOT have a 'sequence' column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()

            tsv_path = Path(tmpdir) / "metadata.tsv"
            _make_metadata_tsv(tsv_path, [("P12345", 100)])
            _make_pdb_files(source_dir, ["P12345"])

            selector = SwissProtDataSelector(
                data_dir=str(data_dir),
                source_dir=str(source_dir),
                metadata_tsv=str(tsv_path),
            )
            df = selector.create_dataset()
            assert "sequence" not in df.columns

    def test_inherits_pdb_data_selector(self):
        """SwissProtDataSelector is a subclass of PDBDataSelector."""
        from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSelector
        assert issubclass(SwissProtDataSelector, PDBDataSelector)
