# SwissProt DataModule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SwissProt data pipeline that reads AlphaFold PDB files, extracts pLDDT from B-factors, and produces training-ready `.pt` files — without modifying existing PDB/Boltz code paths.

**Architecture:** Subclass `PDBDataSelector` → `SwissProtDataSelector` (metadata-based filtering, no PDBManager/RCSB). Subclass `QualityGraftDataModule` → `SwissProtDataModule` (single-pass processing, pLDDT from B-factors, no Boltz). Branch in `build_data_module()` on `database` config key.

**Tech Stack:** PyTorch, PyG (torch_geometric), pandas, graphein (`protein_to_pyg`), Hydra configs, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-swissprot-datamodule-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/quality_graft/data/swissprot_selector.py` | **Create** | `SwissProtDataSelector` — filter UniProt TSV metadata, cross-reference filesystem |
| `src/quality_graft/data/swissprot_datamodule.py` | **Create** | `SwissProtDataModule` — single-pass prepare_data, pLDDT from B-factors |
| `scripts/copy_swissprot.py` | **Create** | Idempotent copy script with filtering |
| `scripts/download_uniprot_tsv.py` | **Create** | One-time UniProt metadata download |
| `configs/data/swissprot.yaml` | **Create** | Hydra config for SwissProt data path |
| `scripts/train.py` | **Modify** (lines 69-112) | Branch `build_data_module()` on `database` field |
| `src/quality_graft/data/__init__.py` | **Modify** | Export new classes |
| `tests/test_swissprot_selector.py` | **Create** | Unit tests for selector |
| `tests/test_swissprot_datamodule.py` | **Create** | Unit tests for datamodule |
| `tests/test_train_script.py` | **Modify** | Add SwissProt build_data_module test |

---

### Task 1: SwissProtDataSelector

**Files:**
- Create: `src/quality_graft/data/swissprot_selector.py`
- Test: `tests/test_swissprot_selector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_swissprot_selector.py`:

```python
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
            # Verify naming convention
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
            # Allow some variance from random sampling
            assert 30 <= len(df) <= 70

    def test_no_chain_column(self):
        """DataFrame should NOT have a 'chain' column (single-chain AlphaFold structures)."""
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
        """DataFrame should NOT have a 'sequence' column (no seq-sim splitting)."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_swissprot_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quality_graft.data.swissprot_selector'`

- [ ] **Step 3: Implement SwissProtDataSelector**

Create `src/quality_graft/data/swissprot_selector.py`:

```python
"""SwissProtDataSelector — metadata-based filtering for AlphaFold SwissProt PDB files.

Unlike PDBDataSelector which queries RCSB via PDBManager, this selector works
entirely from a pre-downloaded UniProt metadata TSV and a directory of PDB files.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

import pandas as pd
from loguru import logger

from la_proteina.proteinfoundation.datasets.pdb_data import PDBDataSelector


class SwissProtDataSelector(PDBDataSelector):
    """Select AlphaFold SwissProt structures by metadata filtering + filesystem check.

    Parameters
    ----------
    source_dir : str
        Path to shared SwissProt PDB directory (e.g. /mnt/labs/shared/databases/swissprot_pdb_v4/files).
    metadata_tsv : str
        Path to UniProt TSV file with accession and length columns.
    alphafold_version : int
        AlphaFold model version for filename pattern (default 4).
    """

    def __init__(
        self,
        data_dir: str,
        source_dir: str,
        metadata_tsv: str,
        alphafold_version: int = 4,
        fraction: float = 1.0,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exclude_ids: Optional[List[str]] = None,
        exclude_ids_from_file: Optional[str] = None,
        num_workers: int = 32,
    ):
        super().__init__(
            data_dir=data_dir,
            fraction=fraction,
            min_length=min_length,
            max_length=max_length,
            exclude_ids=exclude_ids,
            exclude_ids_from_file=exclude_ids_from_file,
            num_workers=num_workers,
            molecule_type=None,
            experiment_types=None,
            oligomeric_min=None,
            oligomeric_max=None,
            best_resolution=None,
            worst_resolution=None,
            has_ligands=None,
            remove_ligands=None,
            remove_non_standard_residues=False,
            remove_pdb_unavailable=False,
            labels=None,
            remove_cath_unavailable=False,
        )
        self.database = "swissprot"
        self.source_dir = pathlib.Path(source_dir)
        self.metadata_tsv = pathlib.Path(metadata_tsv)
        self.alphafold_version = alphafold_version

    def create_dataset(self) -> pd.DataFrame:
        """Filter SwissProt structures by metadata and filesystem presence.

        Returns
        -------
        pd.DataFrame
            Columns: pdb, id, accession, length. No chain or sequence columns.
        """
        if self.df_data is not None:
            return self.df_data

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading UniProt metadata from {}", self.metadata_tsv)
        df = pd.read_csv(self.metadata_tsv, sep="\t")
        logger.info("Loaded {} entries from metadata TSV", len(df))

        # Length filters
        if self.min_length is not None:
            df = df[df["length"] >= self.min_length]
            logger.info("{} entries after min_length={} filter", len(df), self.min_length)
        if self.max_length is not None:
            df = df[df["length"] <= self.max_length]
            logger.info("{} entries after max_length={} filter", len(df), self.max_length)

        # Fraction sampling
        if self.fraction < 1.0:
            df = df.sample(frac=self.fraction)
            logger.info("{} entries after fraction={} sampling", len(df), self.fraction)

        # Exclude IDs
        all_exclude = set()
        if self.exclude_ids:
            all_exclude.update(self.exclude_ids)
        if self.exclude_ids_from_file:
            with open(self.exclude_ids_from_file) as f:
                all_exclude.update(line.strip() for line in f if line.strip())
        if all_exclude:
            df = df[~df["accession"].isin(all_exclude)]
            logger.info("{} entries after excluding {} IDs", len(df), len(all_exclude))

        # Build expected filenames and cross-reference against source_dir
        v = self.alphafold_version
        df["pdb"] = df["accession"].apply(lambda acc: f"AF-{acc}-F1-model_v{v}")
        df["filename"] = df["pdb"] + ".pdb"

        existing_files = set(p.name for p in self.source_dir.iterdir() if p.is_file())
        df = df[df["filename"].isin(existing_files)]
        logger.info("{} entries after filesystem cross-reference", len(df))

        df["id"] = df["pdb"]
        self.df_data = df[["pdb", "id", "accession", "length"]].reset_index(drop=True)
        return self.df_data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_swissprot_selector.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/quality_graft/data/swissprot_selector.py tests/test_swissprot_selector.py
git commit -m "Add SwissProtDataSelector with metadata-based filtering"
```

---

### Task 2: SwissProtDataModule

**Files:**
- Create: `src/quality_graft/data/swissprot_datamodule.py`
- Test: `tests/test_swissprot_datamodule.py`
- Reference: `src/la_proteina/proteinfoundation/datasets/pdb_data.py:628-704` (parent `_load_and_process_pdb`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_swissprot_datamodule.py`:

```python
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
            assert graph.plddt_logits is None  # hard targets only
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_swissprot_datamodule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quality_graft.data.swissprot_datamodule'`

- [ ] **Step 3: Implement SwissProtDataModule**

Create `src/quality_graft/data/swissprot_datamodule.py`:

```python
"""SwissProtDataModule — single-pass processing for AlphaFold SwissProt structures.

Extracts pLDDT from B-factor column (0-100 scale) during PyG conversion.
No Boltz-1 prediction needed. No download step — files are pre-copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger

from openfold.np.residue_constants import resname_to_idx
from graphein.protein.tensor.io import protein_to_pyg
from quality_graft.data.datamodule import QualityGraftDataModule, _save_plddt_status
from quality_graft.data.plddt_utils import plddt_to_bin


class SwissProtDataModule(QualityGraftDataModule):
    """QualityGraftDataModule for AlphaFold SwissProt structures.

    Single-pass processing: pLDDT is extracted from B-factor during PyG
    conversion. No Boltz-1 prediction, no download step.

    Parameters
    ----------
    source_dir : str
        Path to shared SwissProt PDB directory.
    **kwargs
        All remaining arguments forwarded to QualityGraftDataModule.
    """

    def __init__(self, source_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.source_dir = Path(source_dir)

    def _get_file_identifier(self, ds):
        return f"df_swissprot_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}"

    def prepare_data(self):
        """Single-pass preprocessing: PyG conversion with pLDDT from B-factors.

        Does NOT call super().prepare_data() — that would trigger Boltz-1
        prediction from QualityGraftDataModule.
        """
        file_identifier = self._get_file_identifier(self.dataselector)
        df_data_name = f"{file_identifier}.csv"

        if not self.overwrite and (self.data_dir / df_data_name).exists():
            logger.info("{} already exists, skipping processing.", df_data_name)
            return

        df_data = self.dataselector.create_dataset()
        if len(df_data) == 0:
            raise ValueError(
                "SwissProtDataSelector returned 0 structures. "
                "Check metadata_tsv, source_dir, and filter parameters."
            )

        logger.info("Processing {} SwissProt structures.", len(df_data))

        # Process structures (chains=None for single-chain AlphaFold)
        self._process_structure_data(df_data["pdb"].tolist(), chains=None)

        # Save filtered DataFrame
        logger.info("Saving dataset CSV to {}", df_data_name)
        df_data.to_csv(self.data_dir / df_data_name, index=False)

        # Write plddt_status.csv from successfully created .pt files
        plddt_status = {}
        for pt_file in self.processed_dir.glob("*.pt"):
            plddt_status[pt_file.stem] = True

        _save_plddt_status(self.plddt_status_path, plddt_status)

        n_success = len(plddt_status)
        n_failed = len(df_data) - n_success
        logger.info(
            "SwissProt prepare_data complete: {} processed, {} failed.",
            n_success, n_failed,
        )

    def _load_and_process_pdb(
        self, index_pdb_tuple: Union[Tuple[int, str], Tuple[int, str, str]]
    ) -> Optional[str]:
        """Load PDB, convert to PyG graph, extract pLDDT from B-factor.

        Copies the parent method body from PDBLightningDataModule._load_and_process_pdb
        (pdb_data.py lines ~628-704) to avoid double I/O at 550K scale. The only
        additions are pLDDT extraction from B-factor and database tagging.

        If the parent method in pdb_data.py changes, this copy may silently diverge.
        """
        try:
            if len(index_pdb_tuple) == 3:
                i, pdb, chains = index_pdb_tuple
            elif len(index_pdb_tuple) == 2:
                i, pdb = index_pdb_tuple
                chains = "all"
            else:
                raise ValueError("index_pdb_tuple must have 2 or 3 elements")

            path = self.raw_dir / f"{pdb}.{self.format}"
            if path.exists():
                path = str(path)
            elif path.with_suffix("." + self.format + ".gz").exists():
                path = str(path.with_suffix("." + self.format + ".gz"))
            else:
                raise FileNotFoundError(
                    f"{pdb} not found in raw directory. "
                    f"Are you sure it's downloaded and has the format {self.format}?"
                )

            fill_value_coords = 1e-5
            graph = protein_to_pyg(
                path=path,
                chain_selection=chains,
                keep_insertions=True,
                store_het=self.store_het,
                store_bfactor=self.store_bfactor,
                fill_value_coords=fill_value_coords,
            )

        except Exception as e:
            logger.warning("Error processing {} {}: {}", pdb, chains, e)
            return None

        fname = f"{pdb}.pt" if chains == "all" else f"{pdb}_{chains}.pt"

        graph.id = fname.split(".")[0]
        coord_mask = graph.coords != fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        graph.residue_type = torch.tensor(
            [resname_to_idx[residue] for residue in graph.residues]
        ).long()
        graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

        # --- SwissProt additions: pLDDT from B-factor ---
        graph.plddt = graph.bfactor_avg / 100.0       # B-factor is pLDDT on 0-100 scale
        graph.plddt_bin = plddt_to_bin(graph.plddt)    # bin to 0..49
        graph.plddt_logits = None                      # hard targets only
        graph.database = "swissprot"

        if self.pre_transform:
            graph = self.pre_transform(graph)

        if self.pre_filter:
            if self.pre_filter(graph) is not True:
                return None

        torch.save(graph, self.processed_dir / fname)
        return fname
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_swissprot_datamodule.py -v`
Expected: All tests PASS

Note: Some tests depend on `protein_to_pyg` being able to parse the minimal PDB files. If the minimal PDB format causes parse failures, adjust `_make_pdb_file` to produce valid ATOM records that graphein can handle, or mock `protein_to_pyg` in those tests.

- [ ] **Step 5: Commit**

```bash
git add src/quality_graft/data/swissprot_datamodule.py tests/test_swissprot_datamodule.py
git commit -m "Add SwissProtDataModule with B-factor pLDDT extraction"
```

---

### Task 3: Hydra Config + train.py Integration

**Files:**
- Create: `configs/data/swissprot.yaml`
- Modify: `scripts/train.py:69-112`
- Modify: `src/quality_graft/data/__init__.py`
- Modify: `tests/test_train_script.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train_script.py`:

```python
class TestBuildDataModuleSwissProt:
    """Test that build_data_module handles database='swissprot'."""

    def test_swissprot_branch(self):
        """build_data_module returns SwissProtDataModule when database=swissprot."""
        import tempfile
        from omegaconf import OmegaConf
        from quality_graft.data.swissprot_datamodule import SwissProtDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({
                "data": {
                    "database": "swissprot",
                    "data_dir": tmpdir,
                    "source_dir": tmpdir,
                    "metadata_tsv": f"{tmpdir}/metadata.tsv",
                    "alphafold_version": 4,
                    "fraction": 1.0,
                    "min_length": 30,
                    "max_length": 512,
                    "exclude_ids": None,
                    "exclude_ids_from_file": None,
                    "selector_num_workers": 1,
                    "train_val_test": [0.8, 0.15, 0.05],
                    "format": "pdb",
                    "num_plddt_bins": 50,
                    "batch_size": 2,
                    "num_workers": 0,
                    "split_type": "random",
                },
                "training": {
                    "max_length": 512,
                    "min_length": 30,
                    "batch_size": 2,
                    "num_workers": 0,
                },
            })

            # Import from train.py context
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.train import build_data_module

            dm = build_data_module(cfg)
            assert isinstance(dm, SwissProtDataModule)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_script.py::TestBuildDataModuleSwissProt -v`
Expected: FAIL (build_data_module doesn't handle "swissprot" yet)

- [ ] **Step 3: Create the Hydra config**

Create `configs/data/swissprot.yaml`:

```yaml
# SwissProt AlphaFold data configuration
data_dir: data/swissprot/
source_dir: /mnt/labs/shared/databases/swissprot_pdb_v4/files
metadata_tsv: data/swissprot/uniprot_metadata.tsv
alphafold_version: 4
max_length: ${training.max_length}
min_length: ${training.min_length}
format: pdb
local_only: false
num_plddt_bins: 50
train_val_test: [0.8, 0.15, 0.05]
batch_size: ${training.batch_size}
num_workers: ${training.num_workers}
fraction: 1.0
exclude_ids: null
exclude_ids_from_file: null
selector_num_workers: 32
split_type: random  # sequence_similarity not supported (no sequence column)
database: swissprot
```

- [ ] **Step 4: Modify build_data_module in train.py**

In `scripts/train.py`, add imports near the top (after existing imports from quality_graft):

```python
from quality_graft.data.swissprot_selector import SwissProtDataSelector
from quality_graft.data.swissprot_datamodule import SwissProtDataModule
```

Replace the `build_data_module` function (lines 69-112) with:

```python
def build_data_module(cfg: DictConfig):
    """Build the data module from Hydra config."""
    data_cfg = cfg.data
    database = data_cfg.get("database", "pdb")

    if database == "swissprot":
        return _build_swissprot_data_module(data_cfg)
    else:
        return _build_pdb_data_module(cfg)


def _build_swissprot_data_module(data_cfg: DictConfig) -> SwissProtDataModule:
    """Build SwissProtDataModule from config."""
    split_type = data_cfg.get("split_type", "random")
    if split_type != "random":
        raise ValueError(
            f"SwissProt only supports split_type='random', got '{split_type}'. "
            "Sequence-similarity splitting requires a sequence column."
        )

    dataselector = SwissProtDataSelector(
        data_dir=data_cfg.data_dir,
        source_dir=data_cfg.source_dir,
        metadata_tsv=data_cfg.metadata_tsv,
        alphafold_version=data_cfg.get("alphafold_version", 4),
        fraction=data_cfg.get("fraction", 1.0),
        min_length=data_cfg.min_length,
        max_length=data_cfg.max_length,
        exclude_ids=data_cfg.get("exclude_ids", None),
        exclude_ids_from_file=data_cfg.get("exclude_ids_from_file", None),
        num_workers=data_cfg.get("selector_num_workers", 32),
    )
    datasplitter = PDBDataSplitter(
        data_dir=data_cfg.data_dir,
        train_val_test=list(data_cfg.train_val_test),
    )
    transforms = [
        TransformWrapper(lp_transforms.CoordsToNanometers),
        TransformWrapper(lp_transforms.CenterStructureTransform),
    ]
    return SwissProtDataModule(
        data_dir=data_cfg.data_dir,
        source_dir=data_cfg.source_dir,
        dataselector=dataselector,
        datasplitter=datasplitter,
        format="pdb",
        boltz_config={},
        num_plddt_bins=data_cfg.num_plddt_bins,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        transforms=transforms,
    )


def _build_pdb_data_module(cfg: DictConfig) -> QualityGraftDataModule:
    """Build QualityGraftDataModule for PDB data (existing path)."""
    data_cfg = cfg.data

    if data_cfg.get("local_only", False):
        dataselector = None
    else:
        dataselector = PDBDataSelector(
            data_dir=data_cfg.data_dir,
            fraction=data_cfg.get("fraction", 1.0),
            max_length=data_cfg.max_length,
            min_length=data_cfg.min_length,
            molecule_type=data_cfg.molecule_type,
            experiment_types=data_cfg.get("experiment_types", None),
            oligomeric_min=data_cfg.oligomeric_min,
            oligomeric_max=data_cfg.oligomeric_max,
            worst_resolution=data_cfg.get("worst_resolution", None),
            best_resolution=data_cfg.get("best_resolution", None),
            num_workers=data_cfg.get("selector_num_workers", 32),
        )
    datasplitter = PDBDataSplitter(
        data_dir=data_cfg.data_dir,
        train_val_test=list(data_cfg.train_val_test),
    )

    boltz_config = OmegaConf.to_container(data_cfg.boltz, resolve=True)

    transforms = [
        TransformWrapper(lp_transforms.CoordsToNanometers),
        TransformWrapper(lp_transforms.CenterStructureTransform),
    ]

    return QualityGraftDataModule(
        data_dir=data_cfg.data_dir,
        dataselector=dataselector,
        datasplitter=datasplitter,
        format=data_cfg.format,
        boltz_config=boltz_config,
        num_plddt_bins=data_cfg.num_plddt_bins,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        transforms=transforms,
        local_only=data_cfg.get("local_only", False),
    )
```

- [ ] **Step 5: Update __init__.py**

In `src/quality_graft/data/__init__.py`, add exports:

```python
"""Data pipeline for Quality-Graft."""

from quality_graft.data.swissprot_selector import SwissProtDataSelector
from quality_graft.data.swissprot_datamodule import SwissProtDataModule
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_train_script.py -v && pytest tests/test_swissprot_selector.py tests/test_swissprot_datamodule.py -v`
Expected: All PASS

- [ ] **Step 7: Run the full existing test suite**

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: All existing tests still PASS (no regressions in PDB path)

- [ ] **Step 8: Commit**

```bash
git add configs/data/swissprot.yaml scripts/train.py src/quality_graft/data/__init__.py tests/test_train_script.py
git commit -m "Add SwissProt config and train.py build_data_module branching"
```

---

### Task 4: Copy Script

**Files:**
- Create: `scripts/copy_swissprot.py`

- [ ] **Step 1: Implement the copy script**

Create `scripts/copy_swissprot.py`:

```python
#!/usr/bin/env python
"""Copy filtered SwissProt PDB files from shared storage to scratch.

Idempotent: re-running only copies files that don't exist in dest-dir.

Usage:
    python scripts/copy_swissprot.py \
      --source-dir /mnt/labs/shared/databases/swissprot_pdb_v4/files \
      --dest-dir /scratch/schekmenev/swissprot_v4/raw \
      --metadata-tsv data/swissprot/uniprot_metadata.tsv \
      --min-length 30 \
      --max-length 512
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure project paths are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LA_PROTEINA_DIR = SRC_DIR / "la_proteina"
for p in [PROJECT_ROOT, SRC_DIR, LA_PROTEINA_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quality_graft.data.swissprot_selector import SwissProtDataSelector


def main():
    parser = argparse.ArgumentParser(description="Copy filtered SwissProt PDB files to scratch.")
    parser.add_argument("--source-dir", required=True, help="Source SwissProt PDB directory")
    parser.add_argument("--dest-dir", required=True, help="Destination raw/ directory")
    parser.add_argument("--metadata-tsv", required=True, help="UniProt metadata TSV path")
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exclude-ids-file", default=None)
    parser.add_argument("--alphafold-version", type=int, default=4)
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary data_dir (selector needs it but we don't use it)
    selector = SwissProtDataSelector(
        data_dir=str(dest_dir.parent),
        source_dir=args.source_dir,
        metadata_tsv=args.metadata_tsv,
        alphafold_version=args.alphafold_version,
        fraction=args.fraction,
        min_length=args.min_length,
        max_length=args.max_length,
        exclude_ids_from_file=args.exclude_ids_file,
    )

    df = selector.create_dataset()
    print(f"Filtered: {len(df)} structures")

    source_dir = Path(args.source_dir)
    existing = set(p.name for p in dest_dir.iterdir() if p.is_file())
    to_copy = []
    for _, row in df.iterrows():
        fname = f"{row['pdb']}.pdb"
        if fname not in existing:
            to_copy.append(fname)

    print(f"Already present: {len(df) - len(to_copy)}")
    print(f"To copy: {len(to_copy)}")

    for fname in tqdm(to_copy, desc="Copying"):
        shutil.copy2(source_dir / fname, dest_dir / fname)

    # Save filtered file list
    ids_path = dest_dir.parent / "filtered_ids.txt"
    ids_path.write_text("\n".join(df["accession"].tolist()) + "\n")
    print(f"Filtered IDs saved to {ids_path}")
    print(f"Done: {len(to_copy)} copied, {len(df)} total filtered")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is syntactically valid**

Run: `python -c "import ast; ast.parse(open('scripts/copy_swissprot.py').read()); print('OK')""`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/copy_swissprot.py
git commit -m "Add idempotent SwissProt copy script"
```

---

### Task 5: UniProt TSV Download Script

**Files:**
- Create: `scripts/download_uniprot_tsv.py`

- [ ] **Step 1: Implement the download script**

Create `scripts/download_uniprot_tsv.py`:

```python
#!/usr/bin/env python
"""Download UniProt reviewed (SwissProt) metadata TSV.

Fetches accession and length for all reviewed entries. Run once manually.

Usage:
    python scripts/download_uniprot_tsv.py --output data/swissprot/uniprot_metadata.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=tsv&query=(reviewed:true)&fields=accession,length"
)


def main():
    parser = argparse.ArgumentParser(description="Download UniProt SwissProt metadata TSV.")
    parser.add_argument(
        "--output",
        default="data/swissprot/uniprot_metadata.tsv",
        help="Output TSV path (default: data/swissprot/uniprot_metadata.tsv)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SwissProt metadata from UniProt...")
    print(f"URL: {UNIPROT_URL}")

    response = requests.get(UNIPROT_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(output, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Count lines (minus header)
    with open(output) as f:
        n_lines = sum(1 for _ in f) - 1
    print(f"Saved {n_lines} entries to {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is syntactically valid**

Run: `python -c "import ast; ast.parse(open('scripts/download_uniprot_tsv.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/download_uniprot_tsv.py
git commit -m "Add UniProt metadata TSV download script"
```

---

### Task 6: Final Integration Verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --ignore=tests/integration`
Expected: All tests PASS — both existing PDB tests and new SwissProt tests

- [ ] **Step 2: Verify no import errors with Hydra config**

Run: `python scripts/train.py --help`
Expected: Clean exit with Hydra help output (verifies config composition still works)

- [ ] **Step 3: Final commit if any fixups needed**

If any tests needed fixing during the integration run, commit the fixes:

```bash
git add -u
git commit -m "Fix integration issues from SwissProt data pipeline"
```
