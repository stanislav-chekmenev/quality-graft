"""Boltz subprocess runner for generating pLDDT ground truth labels.

Runs `boltz predict` as a subprocess and parses the output files
(pLDDT npz arrays and confidence JSON summaries).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _clean_env_for_boltz() -> dict[str, str]:
    """Return a copy of os.environ with PYTHONPATH entries that contain
    our vendored ``src/boltz/`` removed, so the pip-installed ``boltz``
    package is found instead."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if not pythonpath:
        return env
    cleaned = [
        p for p in pythonpath.split(os.pathsep)
        if not (Path(p) / "boltz").is_dir()
    ]
    env["PYTHONPATH"] = os.pathsep.join(cleaned)
    return env


@dataclass
class BoltzResult:
    """Result of a single Boltz prediction run."""

    pdb_id: str
    plddt: np.ndarray | None  # [N_total] float, 0-1 scale
    confidence_json: dict | None  # Full confidence summary
    success: bool
    error_msg: str | None


def build_boltz_command(
    yaml_path: Path,
    out_dir: Path,
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    override: bool = False,
) -> list[str]:
    """Build the boltz predict CLI command as a list of strings.

    Args:
        yaml_path: Path to the input YAML file for Boltz.
        out_dir: Directory where Boltz writes prediction outputs.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        override: Whether to override existing results.

    Returns:
        Command as a list of strings suitable for subprocess.run().
    """
    cmd = [
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--model",
        model,
        "--devices",
        str(devices),
        "--accelerator",
        accelerator,
        "--diffusion_samples",
        str(diffusion_samples),
        "--sampling_steps",
        str(sampling_steps),
        "--recycling_steps",
        str(recycling_steps),
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if override:
        cmd.append("--override")

    return cmd


def _find_boltz_output(boltz_out_dir: Path, pdb_id: str, filename: str) -> Path | None:
    """Locate a Boltz output file across known directory layouts.

    Boltz writes predictions to a nested directory structure that varies
    by version. This function checks three known layouts:
      1. boltz_out_dir/predictions/{pdb_id}/{filename}
      2. boltz_out_dir/{pdb_id}/predictions/{pdb_id}/{filename}
      3. boltz_out_dir/boltz_results_{pdb_id}/predictions/{pdb_id}/{filename}

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        pdb_id: PDB identifier (stem of the input YAML).
        filename: Name of the file to locate.

    Returns:
        Path to the file, or None if not found.
    """
    candidates = [
        boltz_out_dir / "predictions" / pdb_id / filename,
        boltz_out_dir / pdb_id / "predictions" / pdb_id / filename,
        boltz_out_dir / f"boltz_results_{pdb_id}" / "predictions" / pdb_id / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def find_plddt_npz(boltz_out_dir: Path, pdb_id: str) -> Path | None:
    """Locate the pLDDT npz file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        pdb_id: PDB identifier (stem of the input YAML).

    Returns:
        Path to the npz file, or None if not found.
    """
    return _find_boltz_output(boltz_out_dir, pdb_id, f"plddt_{pdb_id}_model_0.npz")


def find_confidence_json(boltz_out_dir: Path, pdb_id: str) -> Path | None:
    """Locate the confidence JSON file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        pdb_id: PDB identifier (stem of the input YAML).

    Returns:
        Path to the JSON file, or None if not found.
    """
    return _find_boltz_output(
        boltz_out_dir, pdb_id, f"confidence_{pdb_id}_model_0.json"
    )


def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    override: bool = False,
) -> BoltzResult:
    """Run boltz predict as a subprocess and parse results.

    Args:
        yaml_path: Path to the input YAML file. The stem is used as pdb_id.
        out_dir: Directory where Boltz writes prediction outputs.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        override: Whether to override existing results.

    Returns:
        BoltzResult with pLDDT array on success, or error information on failure.
    """
    pdb_id = yaml_path.stem
    cmd = build_boltz_command(
        yaml_path,
        out_dir,
        model,
        devices,
        accelerator,
        diffusion_samples,
        sampling_steps,
        recycling_steps,
        use_msa_server,
        override,
    )

    logger.info("Running Boltz: %s", " ".join(cmd))

    try:
        env = _clean_env_for_boltz()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

        if result.returncode != 0:
            error_msg = (
                f"Boltz failed with return code {result.returncode}\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )
            logger.error(error_msg)
            return BoltzResult(
                pdb_id=pdb_id,
                plddt=None,
                confidence_json=None,
                success=False,
                error_msg=error_msg,
            )

        # Find and load pLDDT
        npz_path = find_plddt_npz(out_dir, pdb_id)
        if npz_path is None:
            return BoltzResult(
                pdb_id=pdb_id,
                plddt=None,
                confidence_json=None,
                success=False,
                error_msg=f"pLDDT npz not found in {out_dir}",
            )

        plddt = np.load(npz_path)["plddt"]

        # Try to load confidence JSON
        conf_json = None
        json_path = find_confidence_json(out_dir, pdb_id)
        if json_path is not None:
            with open(json_path) as f:
                conf_json = json.load(f)

        return BoltzResult(
            pdb_id=pdb_id,
            plddt=plddt,
            confidence_json=conf_json,
            success=True,
            error_msg=None,
        )

    except Exception as e:
        return BoltzResult(
            pdb_id=pdb_id,
            plddt=None,
            confidence_json=None,
            success=False,
            error_msg=str(e),
        )
