"""Tests for boltz_runner batch functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import numpy as np

from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult, run_boltz_predict_dir


class TestBoltzBatchResult:
    """Test BoltzBatchResult dataclass."""

    def test_construction(self):
        result = BoltzBatchResult(
            results={},
            n_submitted=5,
            returncode=0,
            error_msg=None,
        )
        assert result.n_submitted == 5
        assert result.returncode == 0
        assert result.results == {}
        assert result.error_msg is None

    def test_with_results(self):
        import numpy as np
        br = BoltzResult(
            structure_id="1ubq_A",
            plddt=np.array([0.8, 0.9]),
            confidence_json=None,
            success=True,
            error_msg=None,
        )
        result = BoltzBatchResult(
            results={"1ubq_A": br},
            n_submitted=3,
            returncode=0,
            error_msg=None,
        )
        assert len(result.results) == 1
        assert result.results["1ubq_A"].success is True


class TestRunBoltzPredictDir:
    """Test run_boltz_predict_dir()."""

    def test_successful_batch(self, tmp_path):
        """All structures processed successfully."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq_A.yaml").write_text("dummy")
        (input_dir / "2abc_B.yaml").write_text("dummy")

        structure_ids = ["1ubq_A", "2abc_B"]

        for sid in structure_ids:
            pred_dir = out_dir / "predictions" / sid
            pred_dir.mkdir(parents=True)
            np.savez(pred_dir / f"plddt_{sid}_model_0.npz", plddt=np.array([0.8, 0.9]))

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert isinstance(result, BoltzBatchResult)
        assert result.n_submitted == 2
        assert result.returncode == 0
        assert result.error_msg is None
        assert len(result.results) == 2
        assert result.results["1ubq_A"].success is True
        assert result.results["2abc_B"].success is True
        np.testing.assert_array_equal(result.results["1ubq_A"].plddt, [0.8, 0.9])

    def test_partial_results_on_crash(self, tmp_path):
        """Boltz crashes mid-run, but some outputs exist."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq_A.yaml").write_text("dummy")
        (input_dir / "2abc_B.yaml").write_text("dummy")

        structure_ids = ["1ubq_A", "2abc_B"]

        pred_dir = out_dir / "predictions" / "1ubq_A"
        pred_dir.mkdir(parents=True)
        np.savez(pred_dir / "plddt_1ubq_A_model_0.npz", plddt=np.array([0.7]))

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Some error"
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert result.returncode == 1
        assert result.error_msg is not None
        assert len(result.results) == 1
        assert "1ubq_A" in result.results
        assert "2abc_B" not in result.results

    def test_oom_detection(self, tmp_path):
        """OOM errors produce specific error message."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq_A.yaml").write_text("dummy")
        structure_ids = ["1ubq_A"]

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert result.returncode == 1
        assert "OOM" in result.error_msg
        assert "GPU memory exhaustion" in result.error_msg

    def test_directory_mode_layout(self, tmp_path):
        """Boltz directory mode creates boltz_results_{input_dir_name}/ layout."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        (input_dir / "1ubq.yaml").write_text("dummy")
        (input_dir / "2gb1.yaml").write_text("dummy")

        structure_ids = ["1ubq", "2gb1"]

        # Simulate real Boltz directory-mode output layout:
        # out_dir/boltz_results_inputs/predictions/{sid}/plddt_{sid}_model_0.npz
        for sid in structure_ids:
            pred_dir = out_dir / "boltz_results_inputs" / "predictions" / sid
            pred_dir.mkdir(parents=True)
            np.savez(pred_dir / f"plddt_{sid}_model_0.npz", plddt=np.array([0.85, 0.92]))

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""
        mock_proc.stdout = ""

        with patch("quality_graft.data.boltz_runner.subprocess.run", return_value=mock_proc):
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=structure_ids,
            )

        assert result.n_submitted == 2
        assert result.returncode == 0
        assert len(result.results) == 2
        assert result.results["1ubq"].success is True
        assert result.results["2gb1"].success is True
        np.testing.assert_array_equal(result.results["1ubq"].plddt, [0.85, 0.92])

    def test_empty_directory(self, tmp_path):
        """No structures submitted skips subprocess entirely."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        with patch("quality_graft.data.boltz_runner.subprocess.run") as mock_run:
            result = run_boltz_predict_dir(
                input_dir=input_dir,
                out_dir=out_dir,
                structure_ids=[],
            )
            mock_run.assert_not_called()

        assert result.n_submitted == 0
        assert len(result.results) == 0
        assert result.returncode == 0
