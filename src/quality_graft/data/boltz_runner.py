"""Boltz subprocess runner for generating pLDDT ground truth labels.

Runs `boltz predict` as a subprocess and parses the output files
(pLDDT npz arrays and confidence JSON summaries).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

import numpy as np

# Path to the wrapper script that patches Boltz to save logits.
_WRAPPER_SCRIPT = Path(__file__).parent / "boltz_predict_wrapper.py"



def _clean_env_for_boltz() -> dict[str, str]:
    """Return a copy of os.environ with PYTHONPATH entries that contain
    our vendored ``src/boltz/`` removed, so the pip-installed ``boltz``
    package is found instead.  Also ensures the pip-installed NVIDIA
    cuBLAS libraries are on LD_LIBRARY_PATH so cuequivariance_ops can
    find them."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        cleaned = [
            p for p in pythonpath.split(os.pathsep)
            if not (Path(p) / "boltz").is_dir()
        ]
        env["PYTHONPATH"] = os.pathsep.join(cleaned)

    # cuequivariance_ops links against CUDA 12 libs (libcublas.so.12,
    # libnvrtc.so.12) which live in pip nvidia-* packages, not on the
    # default library path.  Add all nvidia lib dirs we can find.
    try:
        import nvidia
        nvidia_root = Path(nvidia.__path__[0])
        lib_dirs = sorted({str(p.parent) for p in nvidia_root.rglob("lib/*.so*")})
        if lib_dirs:
            ld_path = env.get("LD_LIBRARY_PATH", "")
            extra = os.pathsep.join(d for d in lib_dirs if d not in ld_path)
            if extra:
                env["LD_LIBRARY_PATH"] = extra + (os.pathsep + ld_path if ld_path else "")
    except ImportError:
        pass

    return env


@dataclass
class BoltzResult:
    """Result of a single Boltz prediction run."""

    structure_id: str
    plddt: np.ndarray | None  # [N_total] float, 0-1 scale
    plddt_logits: np.ndarray | None  # [N_total, 50] float, raw logits
    pde_logits: np.ndarray | None  # [N_total, N_total, 64] float, raw logits
    confidence_json: dict | None  # Full confidence summary
    success: bool
    error_msg: str | None


@dataclass
class BoltzBatchResult:
    """Result of a batch Boltz prediction run on a directory of YAMLs."""

    results: dict[str, BoltzResult]  # structure_id -> result, for outputs found
    n_submitted: int  # number of YAMLs in the input directory
    returncode: int  # subprocess exit code
    error_msg: str | None  # stderr summary if non-zero exit


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
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
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
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        Command as a list of strings suitable for subprocess.run().
    """
    cmd = [
        sys.executable,
        str(_WRAPPER_SCRIPT),
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
        "--num_workers",
        str(num_workers),
    ]

    if preprocessing_threads is not None:
        cmd.extend(["--preprocessing-threads", str(preprocessing_threads)])

    if max_parallel_samples is not None:
        cmd.extend(["--max_parallel_samples", str(max_parallel_samples)])

    if use_msa_server:
        cmd.append("--use_msa_server")

    if override:
        cmd.append("--override")

    return cmd


def _find_boltz_output(boltz_out_dir: Path, structure_id: str, filename: str) -> Path | None:
    """Locate a Boltz output file across known directory layouts.

    Boltz writes predictions to a nested directory structure that varies
    by version. This function checks three known layouts:
      1. boltz_out_dir/predictions/{structure_id}/{filename}
      2. boltz_out_dir/{structure_id}/predictions/{structure_id}/{filename}
      3. boltz_out_dir/boltz_results_{structure_id}/predictions/{structure_id}/{filename}

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        structure_id: Structure identifier (stem of the input YAML).
        filename: Name of the file to locate.

    Returns:
        Path to the file, or None if not found.
    """
    candidates = [
        boltz_out_dir / "predictions" / structure_id / filename,
        boltz_out_dir / structure_id / "predictions" / structure_id / filename,
        boltz_out_dir / f"boltz_results_{structure_id}" / "predictions" / structure_id / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def find_plddt_npz(boltz_out_dir: Path, structure_id: str) -> Path | None:
    """Locate the pLDDT npz file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        structure_id: Structure identifier (stem of the input YAML).

    Returns:
        Path to the npz file, or None if not found.
    """
    return _find_boltz_output(boltz_out_dir, structure_id, f"plddt_{structure_id}_model_0.npz")


def find_plddt_logits_npz(boltz_out_dir: Path, structure_id: str) -> Path | None:
    """Locate the pLDDT logits npz file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        structure_id: Structure identifier (stem of the input YAML).

    Returns:
        Path to the logits npz file, or None if not found.
    """
    return _find_boltz_output(
        boltz_out_dir, structure_id, f"plddt_logits_{structure_id}_model_0.npz"
    )


def find_pde_logits_npz(boltz_out_dir: Path, structure_id: str) -> Path | None:
    """Locate the PDE logits npz file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        structure_id: Structure identifier (stem of the input YAML).

    Returns:
        Path to the PDE logits npz file, or None if not found.
    """
    return _find_boltz_output(
        boltz_out_dir, structure_id, f"pde_logits_{structure_id}_model_0.npz"
    )


def find_confidence_json(boltz_out_dir: Path, structure_id: str) -> Path | None:
    """Locate the confidence JSON file in Boltz output directory.

    Args:
        boltz_out_dir: Root output directory passed to Boltz.
        structure_id: Structure identifier (stem of the input YAML).

    Returns:
        Path to the JSON file, or None if not found.
    """
    return _find_boltz_output(
        boltz_out_dir, structure_id, f"confidence_{structure_id}_model_0.json"
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
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
) -> BoltzResult:
    """Run boltz predict as a subprocess and parse results.

    Args:
        yaml_path: Path to the input YAML file. The stem is used as structure_id.
        out_dir: Directory where Boltz writes prediction outputs.
        model: Model name (default: "boltz1").
        devices: Number of devices to use.
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        override: Whether to override existing results.
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        BoltzResult with pLDDT array on success, or error information on failure.
    """
    structure_id = yaml_path.stem
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
        num_workers,
        preprocessing_threads,
        max_parallel_samples,
    )

    logger.info("Running Boltz: {}", " ".join(cmd))

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
                structure_id=structure_id,
                plddt=None,
                plddt_logits=None,
                pde_logits=None,
                confidence_json=None,
                success=False,
                error_msg=error_msg,
            )

        # Find and load pLDDT
        npz_path = find_plddt_npz(out_dir, structure_id)
        if npz_path is None:
            return BoltzResult(
                structure_id=structure_id,
                plddt=None,
                plddt_logits=None,
                pde_logits=None,
                confidence_json=None,
                success=False,
                error_msg=f"pLDDT npz not found in {out_dir}",
            )

        plddt = np.load(npz_path)["plddt"]

        # Try to load pLDDT logits
        plddt_logits = None
        logits_npz_path = find_plddt_logits_npz(out_dir, structure_id)
        if logits_npz_path is not None:
            plddt_logits = np.load(logits_npz_path)["plddt_logits"]

        # Try to load PDE logits
        pde_logits = None
        pde_logits_npz_path = find_pde_logits_npz(out_dir, structure_id)
        if pde_logits_npz_path is not None:
            pde_logits = np.load(pde_logits_npz_path)["pde_logits"]

        # Try to load confidence JSON
        conf_json = None
        json_path = find_confidence_json(out_dir, structure_id)
        if json_path is not None:
            with open(json_path) as f:
                conf_json = json.load(f)

        return BoltzResult(
            structure_id=structure_id,
            plddt=plddt,
            plddt_logits=plddt_logits,
            pde_logits=pde_logits,
            confidence_json=conf_json,
            success=True,
            error_msg=None,
        )

    except Exception as e:
        return BoltzResult(
            structure_id=structure_id,
            plddt=None,
            plddt_logits=None,
            pde_logits=None,
            confidence_json=None,
            success=False,
            error_msg=str(e),
        )


def run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    structure_ids: list[str],
    model: str = "boltz1",
    devices: int = 1,
    accelerator: str = "gpu",
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    timeout: int | None = None,
    num_workers: int = 2,
    preprocessing_threads: int | None = None,
    max_parallel_samples: int | None = None,
) -> BoltzBatchResult:
    """Run boltz predict on a directory of YAMLs and collect results.

    Passes the entire input_dir to a single `boltz predict` invocation
    with native multi-GPU support via --devices. After the subprocess
    finishes (or crashes), iterates over structure_ids and collects
    whatever pLDDT outputs exist.

    Args:
        input_dir: Directory containing YAML files for Boltz.
        out_dir: Directory where Boltz writes prediction outputs.
        structure_ids: List of structure IDs to collect results for.
        model: Model name (default: "boltz1").
        devices: Number of devices to use (passed as --devices to Boltz).
        accelerator: Accelerator type ("gpu" or "cpu").
        diffusion_samples: Number of diffusion samples.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        use_msa_server: Whether to use the MSA server.
        timeout: Max seconds to wait for the subprocess. None means no limit.
        num_workers: Number of Boltz dataloader workers.
        preprocessing_threads: Number of Boltz preprocessing threads (None = Boltz default).
        max_parallel_samples: Max diffusion samples processed in parallel (None = Boltz default of 5).

    Returns:
        BoltzBatchResult with per-structure results for outputs found.
    """
    n_submitted = len(structure_ids)

    if n_submitted == 0:
        logger.info("No structures to process, skipping Boltz subprocess.")
        return BoltzBatchResult(results={}, n_submitted=0, returncode=0, error_msg=None)

    cmd = build_boltz_command(
        yaml_path=input_dir,
        out_dir=out_dir,
        model=model,
        devices=devices,
        accelerator=accelerator,
        diffusion_samples=diffusion_samples,
        sampling_steps=sampling_steps,
        recycling_steps=recycling_steps,
        use_msa_server=use_msa_server,
        override=False,
        num_workers=num_workers,
        preprocessing_threads=preprocessing_threads,
        max_parallel_samples=max_parallel_samples,
    )

    logger.info("Running Boltz on directory ({} structures): {}", n_submitted, " ".join(cmd))

    error_msg = None
    returncode = 0

    try:
        env = _clean_env_for_boltz()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=env,
            timeout=timeout,
        )
        returncode = proc.returncode

        # Always log subprocess output for debuggability
        if proc.stdout.strip():
            logger.debug("Boltz stdout:\n{}", proc.stdout[-2000:])

        if returncode != 0:
            stderr = proc.stderr
            if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
                error_msg = (
                    f"Boltz OOM: GPU memory exhaustion during batch prediction. "
                    f"Re-run to process remaining structures, or reduce max_length / increase GPU memory.\n"
                    f"stderr: {stderr[-500:]}"
                )
                logger.error(error_msg)
            else:
                error_msg = (
                    f"Boltz failed with return code {returncode}\n"
                    f"stderr: {stderr}\nstdout: {proc.stdout}"
                )
                logger.error(error_msg)

    except subprocess.TimeoutExpired as e:
        error_msg = (
            f"Boltz subprocess timed out after {timeout}s. "
            f"Reduce chunk_size or increase timeout."
        )
        returncode = -2
        logger.error(error_msg)

    except Exception as e:
        error_msg = str(e)
        returncode = -1
        logger.error("Boltz subprocess exception: {}", e)

    # Collect results for whatever outputs exist.
    # In directory mode, Boltz nests output under boltz_results_{input_dir_name}/
    # so try that first, then fall back to out_dir directly.
    boltz_results_dir = out_dir / f"boltz_results_{input_dir.name}"
    lookup_dir = boltz_results_dir if boltz_results_dir.is_dir() else out_dir

    if lookup_dir != out_dir:
        logger.debug("Using Boltz results directory: {}", lookup_dir)

    results: dict[str, BoltzResult] = {}
    for sid in structure_ids:
        npz_path = find_plddt_npz(lookup_dir, sid)
        if npz_path is None:
            logger.warning(
                "[{}] pLDDT output not found under {}. "
                "Boltz may have skipped or failed this structure silently.",
                sid, lookup_dir,
            )
            continue

        plddt = np.load(npz_path)["plddt"]

        # Try to load pLDDT logits
        plddt_logits = None
        logits_npz_path = find_plddt_logits_npz(lookup_dir, sid)
        if logits_npz_path is not None:
            plddt_logits = np.load(logits_npz_path)["plddt_logits"]

        # Try to load PDE logits
        pde_logits = None
        pde_logits_npz_path = find_pde_logits_npz(lookup_dir, sid)
        if pde_logits_npz_path is not None:
            pde_logits = np.load(pde_logits_npz_path)["pde_logits"]

        conf_json = None
        json_path = find_confidence_json(lookup_dir, sid)
        if json_path is not None:
            with open(json_path) as f:
                conf_json = json.load(f)

        results[sid] = BoltzResult(
            structure_id=sid,
            plddt=plddt,
            plddt_logits=plddt_logits,
            pde_logits=pde_logits,
            confidence_json=conf_json,
            success=True,
            error_msg=None,
        )

    n_found = len(results)
    if error_msg and n_found > 0:
        error_msg += f"\n{n_found} of {n_submitted} structures completed before failure."

    logger.info(
        "Boltz batch complete: {}/{} structures produced pLDDT (returncode={})",
        n_found, n_submitted, returncode,
    )

    return BoltzBatchResult(
        results=results,
        n_submitted=n_submitted,
        returncode=returncode,
        error_msg=error_msg,
    )
