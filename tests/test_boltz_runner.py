"""Tests for boltz_runner batch functionality."""

from quality_graft.data.boltz_runner import BoltzBatchResult, BoltzResult


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
