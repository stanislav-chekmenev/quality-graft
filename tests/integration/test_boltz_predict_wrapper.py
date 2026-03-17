"""Heavy integration test for the boltz_predict_wrapper.

Runs the wrapper script as a subprocess on a tiny protein (Trp-cage, 20 aa)
and verifies that the patched writer saves plddt_logits, pde_logits, and
resolved_logits npz files alongside the standard Boltz outputs.

Requires: GPU + Boltz checkpoint at ckpt/boltz1_conf.ckpt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# 20-residue Trp-cage miniprotein (PDB 1L2Y, chain A).
# Small enough for fast inference, large enough to be a real protein.
TRPCAGE_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
STRUCTURE_ID = "trpcage"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WRAPPER_SCRIPT = (
    PROJECT_ROOT / "src" / "quality_graft" / "data" / "boltz_predict_wrapper.py"
)

CKPT_PATH = PROJECT_ROOT / "ckpt" / "boltz1_conf.ckpt"


def _make_boltz_yaml(sequence: str, structure_id: str = "A") -> str:
    """Create a minimal single-sequence Boltz input YAML."""
    return (
        f"version: 1\n"
        f"sequences:\n"
        f"  - protein:\n"
        f"      id: {structure_id}\n"
        f"      sequence: {sequence}\n"
        f"      msa: empty\n"
    )


def _clean_env() -> dict[str, str]:
    """Return env with vendored src/boltz/ removed from PYTHONPATH."""
    import os

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        cleaned = [
            p for p in pythonpath.split(os.pathsep)
            if not (Path(p) / "boltz").is_dir()
        ]
        env["PYTHONPATH"] = os.pathsep.join(cleaned)
    return env


@pytest.mark.heavy
class TestBoltzPredictWrapper:
    """End-to-end test: wrapper script → Boltz inference → logit npz files."""

    def test_wrapper_produces_logit_npz_files(self, tmp_path):
        """Run the wrapper on a tiny protein and verify logit outputs."""
        # -- Arrange ----------------------------------------------------------
        input_yaml = tmp_path / f"{STRUCTURE_ID}.yaml"
        input_yaml.write_text(_make_boltz_yaml(TRPCAGE_SEQUENCE))
        out_dir = tmp_path / "boltz_out"
        out_dir.mkdir()

        cmd = [
            sys.executable,
            str(WRAPPER_SCRIPT),
            "predict",
            str(input_yaml),
            "--out_dir", str(out_dir),
            "--checkpoint", str(CKPT_PATH),
            "--model", "boltz1",
            "--accelerator", "gpu",
            "--devices", "1",
            "--diffusion_samples", "1",
            "--sampling_steps", "20",   # fast: fewer steps
            "--recycling_steps", "1",   # fast: fewer recycling
            "--num_workers", "0",
        ]

        # -- Act --------------------------------------------------------------
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env(),
            timeout=600,  # generous: first run downloads ckpt
        )

        # -- Assert: subprocess succeeded -------------------------------------
        assert result.returncode == 0, (
            f"Wrapper failed (rc={result.returncode}).\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}\n"
            f"--- stdout (last 1000 chars) ---\n{result.stdout[-1000:]}"
        )

        # -- Assert: find the Boltz output directory --------------------------
        # Boltz nests under  out_dir/boltz_results_{yaml_stem}/predictions/{id}/
        boltz_results = out_dir / f"boltz_results_{STRUCTURE_ID}"
        assert boltz_results.is_dir(), (
            f"Expected Boltz results dir at {boltz_results}. "
            f"Contents of {out_dir}: {list(out_dir.iterdir())}"
        )

        pred_dir = boltz_results / "predictions" / STRUCTURE_ID
        assert pred_dir.is_dir(), (
            f"Expected predictions dir at {pred_dir}. "
            f"Contents of {boltz_results}: {list(boltz_results.rglob('*'))}"
        )

        # -- Assert: standard pLDDT npz exists (sanity check) ----------------
        plddt_npz = pred_dir / f"plddt_{STRUCTURE_ID}_model_0.npz"
        assert plddt_npz.is_file(), f"Standard pLDDT npz not found at {plddt_npz}"
        plddt = np.load(plddt_npz)["plddt"]
        n_tokens = plddt.shape[0]
        assert n_tokens >= len(TRPCAGE_SEQUENCE), (
            f"pLDDT has {n_tokens} tokens, expected >= {len(TRPCAGE_SEQUENCE)}"
        )

        # -- Assert: pLDDT logits npz ----------------------------------------
        plddt_logits_npz = pred_dir / f"plddt_logits_{STRUCTURE_ID}_model_0.npz"
        assert plddt_logits_npz.is_file(), (
            f"pLDDT logits npz not found at {plddt_logits_npz}. "
            f"Files in pred_dir: {[f.name for f in pred_dir.iterdir()]}"
        )
        plddt_logits = np.load(plddt_logits_npz)["plddt_logits"]
        assert plddt_logits.ndim == 2, f"Expected 2D, got shape {plddt_logits.shape}"
        assert plddt_logits.shape[0] == n_tokens, (
            f"pLDDT logits dim0={plddt_logits.shape[0]} != plddt tokens={n_tokens}"
        )
        assert plddt_logits.shape[1] == 50, (
            f"pLDDT logits dim1={plddt_logits.shape[1]}, expected 50 bins"
        )
        assert np.isfinite(plddt_logits).all(), "pLDDT logits contain non-finite values"

        # -- Assert: PDE logits npz -------------------------------------------
        pde_logits_npz = pred_dir / f"pde_logits_{STRUCTURE_ID}_model_0.npz"
        assert pde_logits_npz.is_file(), (
            f"PDE logits npz not found at {pde_logits_npz}. "
            f"Files in pred_dir: {[f.name for f in pred_dir.iterdir()]}"
        )
        pde_logits = np.load(pde_logits_npz)["pde_logits"]
        assert pde_logits.ndim == 3, f"Expected 3D, got shape {pde_logits.shape}"
        assert pde_logits.shape[0] == n_tokens, (
            f"PDE logits dim0={pde_logits.shape[0]} != plddt tokens={n_tokens}"
        )
        assert pde_logits.shape[1] == n_tokens, (
            f"PDE logits dim1={pde_logits.shape[1]} != plddt tokens={n_tokens}"
        )
        assert pde_logits.shape[2] == 64, (
            f"PDE logits dim2={pde_logits.shape[2]}, expected 64 bins"
        )
        assert np.isfinite(pde_logits).all(), "PDE logits contain non-finite values"

        # -- Assert: resolved logits npz --------------------------------------
        resolved_npz = pred_dir / f"resolved_logits_{STRUCTURE_ID}_model_0.npz"
        assert resolved_npz.is_file(), (
            f"Resolved logits npz not found at {resolved_npz}. "
            f"Files in pred_dir: {[f.name for f in pred_dir.iterdir()]}"
        )
        resolved_logits = np.load(resolved_npz)["resolved_logits"]
        assert resolved_logits.shape == (n_tokens, 2), (
            f"Resolved logits shape {resolved_logits.shape}, expected ({n_tokens}, 2)"
        )

        # -- Assert: boltz_runner finders work with the output ----------------
        from quality_graft.data.boltz_runner import (
            find_plddt_logits_npz,
            find_pde_logits_npz,
            find_plddt_npz,
        )

        assert find_plddt_npz(out_dir, STRUCTURE_ID) is not None
        assert find_plddt_logits_npz(out_dir, STRUCTURE_ID) is not None
        assert find_pde_logits_npz(out_dir, STRUCTURE_ID) is not None
