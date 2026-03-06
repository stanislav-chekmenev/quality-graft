"""Integration tests for the dataset generation pipeline.

These tests require GPU and the Boltz model to be available.
Run with: pytest tests/integration/test_generate_dataset.py -v --run-heavy
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import pytest
import torch


@pytest.mark.heavy
class TestGenerateDataset:
    """End-to-end tests for the dataset generation script."""

    def test_1ubq_end_to_end(self, tmp_path):
        """Run full pipeline on 1ubq.pdb and verify output."""
        pdb_path = PROJECT_ROOT / "data" / "1ubq.pdb"
        if not pdb_path.exists():
            pytest.skip("data/1ubq.pdb not found")

        output_dir = tmp_path / "labels"
        work_dir = tmp_path / "boltz_work"

        result = subprocess.run(
            [
                sys.executable, "scripts/generate_dataset.py",
                "--single-pdb", str(pdb_path),
                "--output-dir", str(output_dir),
                "--work-dir", str(work_dir),
                "--no-wandb",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env={
                **dict(__import__("os").environ),
                "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}",
            },
        )

        assert result.returncode == 0, f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Check output file exists
        pt_file = output_dir / "1ubq.pt"
        assert pt_file.exists(), f"Output file not found. Dir contents: {list(output_dir.iterdir()) if output_dir.exists() else 'dir missing'}"

        # Load and verify
        data = torch.load(pt_file, weights_only=False)

        assert data["pdb_id"] == "1ubq"
        assert "A" in data["sequences"]
        assert data["sequences"]["A"].startswith("MQIFVKTLTG")
        assert data["n_residues"] == 76

        # pLDDT checks
        plddt = data["plddt"]
        assert plddt.shape == (76,), f"Expected shape (76,), got {plddt.shape}"
        assert plddt.dtype == torch.float32
        assert (plddt >= 0).all() and (plddt <= 1).all(), "pLDDT values should be in [0, 1]"

        # Bin checks
        plddt_bin = data["plddt_bin"]
        assert plddt_bin.shape == (76,)
        assert plddt_bin.dtype == torch.int64
        assert (plddt_bin >= 0).all() and (plddt_bin <= 49).all()

        # Chain lengths
        assert data["chain_lengths"]["A"] == 76
