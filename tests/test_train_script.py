"""Smoke tests for scripts/train.py."""

import os
import subprocess
import sys
from pathlib import Path


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