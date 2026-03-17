#!/usr/bin/env python
"""Wrapper around ``boltz predict`` that also saves per-residue logits.

Standard ``boltz predict`` computes pLDDT logits [N, 50], PDE logits
[N, N, 64], and resolved logits [N, 2] inside the confidence head but
discards them — only the aggregated continuous values are saved.

This wrapper monkey-patches two spots before invoking the CLI:

1. ``Boltz1.predict_step`` — includes logits in the prediction dict.
2. ``BoltzWriter.write_on_batch_end`` — saves logits as .npz files
   alongside the standard outputs.

Usage (drop-in replacement for ``boltz predict``):

    python -m quality_graft.data.boltz_predict_wrapper predict \\
        path/to/inputs --out_dir path/to/out --model boltz1 ...

All CLI arguments are forwarded to the upstream ``boltz predict`` command.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Patch 1: Boltz1.predict_step  —  keep logits in the prediction dict
# ---------------------------------------------------------------------------

def _patched_predict_step(self, batch, batch_idx, dataloader_idx=0):
    """Replacement for ``Boltz1.predict_step`` that also captures logits."""
    try:
        out = self(
            batch,
            recycling_steps=self.predict_args["recycling_steps"],
            num_sampling_steps=self.predict_args["sampling_steps"],
            diffusion_samples=self.predict_args["diffusion_samples"],
            max_parallel_samples=self.predict_args["diffusion_samples"],
            run_confidence_sequentially=True,
        )

        pred_dict: dict[str, Tensor | bool] = {"exception": False}
        pred_dict["masks"] = batch["atom_pad_mask"]
        pred_dict["coords"] = out["sample_atom_coords"]
        pred_dict["s"] = out["s"]
        pred_dict["z"] = out["z"]

        if self.predict_args.get("write_confidence_summary", True):
            pred_dict["confidence_score"] = (
                4 * out["complex_plddt"]
                + (
                    out["iptm"]
                    if not torch.allclose(
                        out["iptm"], torch.zeros_like(out["iptm"])
                    )
                    else out["ptm"]
                )
            ) / 5
            for key in [
                "ptm", "iptm", "ligand_iptm", "protein_iptm",
                "pair_chains_iptm", "complex_plddt", "complex_iplddt",
                "complex_pde", "complex_ipde", "plddt",
            ]:
                pred_dict[key] = out[key]

        if self.predict_args.get("write_full_pae", True):
            pred_dict["pae"] = out["pae"]
        if self.predict_args.get("write_full_pde", False):
            pred_dict["pde"] = out["pde"]

        # ---- NEW: also capture the raw logits ----
        for logit_key in ("plddt_logits", "pde_logits", "resolved_logits"):
            if logit_key in out:
                pred_dict[logit_key] = out[logit_key]

        return pred_dict

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("| WARNING: ran out of memory, skipping batch")
            torch.cuda.empty_cache()
            gc.collect()
            return {"exception": True}
        raise


# ---------------------------------------------------------------------------
# Patch 2: BoltzWriter.write_on_batch_end  —  save logits as npz
# ---------------------------------------------------------------------------

_original_write_on_batch_end = None  # set at patch time


def _patched_write_on_batch_end(
    self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx,
):
    """Call the original writer, then additionally save logit npz files."""
    # Run the standard writer first (structures, plddt, pde, confidence JSON…)
    _original_write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices,
        batch, batch_idx, dataloader_idx,
    )

    if prediction["exception"]:
        return

    records = batch["record"]
    pad_masks = prediction["masks"]
    coords = prediction["coords"].unsqueeze(0)

    # Reproduce the ranking logic from the original writer
    if "confidence_score" in prediction:
        argsort = torch.argsort(prediction["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
    else:
        idx_to_rank = {i: i for i in range(len(records))}

    for record, coord, pad_mask in zip(records, coords, pad_masks):
        struct_dir = self.output_dir / record.id
        struct_dir.mkdir(exist_ok=True)

        for model_idx in range(coord.shape[0]):
            rank = idx_to_rank[model_idx]

            # pLDDT logits  [N_padded, 50]  →  save full (matches plddt npz)
            if "plddt_logits" in prediction:
                logits = prediction["plddt_logits"][model_idx]
                path = struct_dir / f"plddt_logits_{record.id}_model_{rank}.npz"
                np.savez_compressed(path, plddt_logits=logits.cpu().numpy())

            # PDE logits  [N_padded, N_padded, 64]
            if "pde_logits" in prediction:
                pde_logits = prediction["pde_logits"][model_idx]
                path = struct_dir / f"pde_logits_{record.id}_model_{rank}.npz"
                np.savez_compressed(path, pde_logits=pde_logits.cpu().numpy())

            # Resolved logits  [N_padded, 2]
            if "resolved_logits" in prediction:
                resolved = prediction["resolved_logits"][model_idx]
                path = struct_dir / f"resolved_logits_{record.id}_model_{rank}.npz"
                np.savez_compressed(path, resolved_logits=resolved.cpu().numpy())


# ---------------------------------------------------------------------------
# Apply patches and delegate to the boltz CLI
# ---------------------------------------------------------------------------

def main():
    global _original_write_on_batch_end

    from boltz.model.models.boltz1 import Boltz1
    from boltz.data.write.writer import BoltzWriter
    from boltz.main import cli

    # Patch predict_step
    Boltz1.predict_step = _patched_predict_step

    # Patch writer
    _original_write_on_batch_end = BoltzWriter.write_on_batch_end
    BoltzWriter.write_on_batch_end = _patched_write_on_batch_end

    # Forward all CLI args to boltz (click parses sys.argv)
    cli()


if __name__ == "__main__":
    main()
