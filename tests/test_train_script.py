"""Smoke tests for scripts/train.py."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


class TestTrainScriptHelp:
    def test_help_flag(self):
        """train.py --help should exit cleanly."""
        env = os.environ.copy()
        # Ensure all needed paths are on PYTHONPATH
        extra_paths = [str(PROJECT_ROOT), str(SRC_DIR), str(SRC_DIR / "la_proteina")]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--help"],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


class TestBuildDataModuleSwissProt:
    """Test that build_data_module handles database='swissprot'."""

    def test_swissprot_branch(self):
        """build_data_module returns SwissProtDataModule when database=swissprot."""
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

            # Import build_data_module from train.py
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.train import build_data_module

            dm = build_data_module(cfg)
            assert isinstance(dm, SwissProtDataModule)