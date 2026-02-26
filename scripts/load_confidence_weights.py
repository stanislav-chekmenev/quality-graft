#!/usr/bin/env python3
"""Load Boltz-1 confidence module weights from the HuggingFace checkpoint.

This script instantiates the Boltz-1 ConfidenceModule with the exact architecture
used during training (imitate_trunk=True, 48-block pairformer, etc.) and loads
the pretrained weights from the `boltz1_conf.ckpt` checkpoint.

Checkpoint source: https://huggingface.co/boltz-community/boltz-1/tree/main
    - boltz1_conf.ckpt (3.6 GB) — confidence module weights (PyTorch Lightning format)

Usage:
    python scripts/load_confidence_weights.py [--ckpt_path PATH] [--device DEVICE]

The checkpoint is a PyTorch Lightning checkpoint with the following structure:
    - state_dict: contains all model keys (6611 total)
    - hyper_parameters: model configuration
    - The confidence module keys are prefixed with "confidence_module."
    - There are 2887 confidence_module.* keys

The confidence module was trained with `imitate_trunk=True`, meaning it has its own:
    - input_embedder (AtomAttentionEncoder)
    - msa_module (4 MSA blocks)
    - pairformer_module (48 pairformer blocks, 16 heads)
    - confidence_heads (pLDDT, PDE, PAE, resolved)
    - Various projection layers (s_init, z_init, recycling, etc.)
"""

import argparse
import torch

from pathlib import Path

from src.boltz.model.modules.confidence import ConfidenceModule


# ──────────────────────────────────────────────────────────────────────
# Architecture constants — extracted from the checkpoint hyper_parameters
# and the confidence.yaml training config.
# ──────────────────────────────────────────────────────────────────────

# Core dimensions
TOKEN_S = 384  # single representation dim
TOKEN_Z = 128  # pair representation dim
ATOM_S = 128   # atom single dim
ATOM_Z = 16    # atom pair dim
ATOM_FEATURE_DIM = 389  # atom feature dim
ATOMS_PER_WINDOW_QUERIES = 32
ATOMS_PER_WINDOW_KEYS = 128

# Confidence model args (from hyper_parameters.confidence_model_args)
CONFIDENCE_MODEL_ARGS = {
    "num_dist_bins": 64,
    "max_dist": 22,
    "add_s_to_z_prod": True,
    "add_s_input_to_s": True,
    "use_s_diffusion": True,
    "add_z_input_to_z": True,
    "confidence_args": {
        "num_plddt_bins": 50,
        "num_pde_bins": 64,
        "num_pae_bins": 64,
    },
}

# Pairformer args (from hyper_parameters.pairformer_args)
PAIRFORMER_ARGS = {
    "num_blocks": 48,
    "num_heads": 16,
    "dropout": 0.25,
    "post_layer_norm": False,
    "activation_checkpointing": False,  # Disabled for inference
    "offload_to_cpu": False,
}

# Embedder args (from hyper_parameters.embedder_args)
EMBEDDER_ARGS = {
    "atom_encoder_depth": 3,
    "atom_encoder_heads": 4,
}

# Full embedder args (constructed the same way as in Boltz1.__init__)
FULL_EMBEDDER_ARGS = {
    "atom_s": ATOM_S,
    "atom_z": ATOM_Z,
    "token_s": TOKEN_S,
    "token_z": TOKEN_Z,
    "atoms_per_window_queries": ATOMS_PER_WINDOW_QUERIES,
    "atoms_per_window_keys": ATOMS_PER_WINDOW_KEYS,
    "atom_feature_dim": ATOM_FEATURE_DIM,
    "no_atom_encoder": False,
    **EMBEDDER_ARGS,
}

# MSA args (from hyper_parameters.msa_args)
MSA_ARGS = {
    "msa_s": 64,
    "msa_blocks": 4,
    "msa_dropout": 0.15,
    "z_dropout": 0.25,
    "pairwise_head_width": 32,
    "pairwise_num_heads": 4,
    "postpone_outer_product": True,
    "activation_checkpointing": False,  # Disabled for inference
    "offload_to_cpu": False,
}


def create_confidence_module() -> ConfidenceModule:
    """Instantiate the ConfidenceModule with the Boltz-1 architecture.

    Returns
    -------
    ConfidenceModule
        The confidence module with randomly initialized weights.

    """
    module = ConfidenceModule(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        compute_pae=True,  # alpha_pae=1 > 0 in the config
        imitate_trunk=True,
        pairformer_args=PAIRFORMER_ARGS,
        full_embedder_args=FULL_EMBEDDER_ARGS,
        msa_args=MSA_ARGS,
        **CONFIDENCE_MODEL_ARGS,
    )
    return module


def load_confidence_weights(
    ckpt_path: str | Path,
    device: str = "cpu",
) -> ConfidenceModule:
    """Load the ConfidenceModule with pretrained weights from the checkpoint.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to the boltz1_conf.ckpt file.
    device : str
        Device to load the weights onto (default: "cpu").

    Returns
    -------
    ConfidenceModule
        The confidence module with loaded pretrained weights.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If there are missing or unexpected keys during weight loading.

    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract confidence module state dict
    full_state_dict = checkpoint["state_dict"]
    prefix = "confidence_module."
    conf_state_dict = {
        k[len(prefix):]: v
        for k, v in full_state_dict.items()
        if k.startswith(prefix)
    }
    print(f"Found {len(conf_state_dict)} confidence module parameters in checkpoint")

    # Create the module
    print("Instantiating ConfidenceModule...")
    module = create_confidence_module()
    module = module.to(device)

    # Load weights
    result = module.load_state_dict(conf_state_dict, strict=True)
    print(f"Weight loading result: {result}")

    # Verify parameter count
    total_params = sum(p.numel() for p in module.parameters())
    total_buffers = sum(b.numel() for b in module.buffers())
    print(f"Total parameters: {total_params:,}")
    print(f"Total buffers: {total_buffers:,}")
    print(f"Confidence module loaded successfully!")

    return module


def print_module_summary(module: ConfidenceModule) -> None:
    """Print a summary of the confidence module architecture.

    Parameters
    ----------
    module : ConfidenceModule
        The loaded confidence module.

    """
    print("\n" + "=" * 70)
    print("CONFIDENCE MODULE SUMMARY")
    print("=" * 70)

    # Count parameters per sub-module
    submodule_params = {}
    for name, param in module.named_parameters():
        top_level = name.split(".")[0]
        if top_level not in submodule_params:
            submodule_params[top_level] = 0
        submodule_params[top_level] += param.numel()

    print(f"\n{'Sub-module':<35} {'Parameters':>15}")
    print("-" * 52)
    for name, count in sorted(submodule_params.items(), key=lambda x: -x[1]):
        print(f"  {name:<33} {count:>15,}")
    print("-" * 52)
    total = sum(submodule_params.values())
    print(f"  {'TOTAL':<33} {total:>15,}")

    # Key architecture details
    print(f"\nArchitecture details:")
    print(f"  imitate_trunk: True")
    print(f"  token_s (single dim): {TOKEN_S}")
    print(f"  token_z (pair dim): {TOKEN_Z}")
    print(f"  pairformer blocks: {PAIRFORMER_ARGS['num_blocks']}")
    print(f"  pairformer heads: {PAIRFORMER_ARGS['num_heads']}")
    print(f"  num_dist_bins: {CONFIDENCE_MODEL_ARGS['num_dist_bins']}")
    print(f"  num_plddt_bins: {CONFIDENCE_MODEL_ARGS['confidence_args']['num_plddt_bins']}")
    print(f"  num_pde_bins: {CONFIDENCE_MODEL_ARGS['confidence_args']['num_pde_bins']}")
    print(f"  num_pae_bins: {CONFIDENCE_MODEL_ARGS['confidence_args']['num_pae_bins']}")
    print(f"  use_s_diffusion: {CONFIDENCE_MODEL_ARGS['use_s_diffusion']}")
    print(f"  add_s_to_z_prod: {CONFIDENCE_MODEL_ARGS['add_s_to_z_prod']}")
    print(f"  add_s_input_to_s: {CONFIDENCE_MODEL_ARGS['add_s_input_to_s']}")
    print(f"  add_z_input_to_z: {CONFIDENCE_MODEL_ARGS['add_z_input_to_z']}")


def main():
    parser = argparse.ArgumentParser(
        description="Load Boltz-1 confidence module weights"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/boltz1_conf.ckpt",
        help="Path to the boltz1_conf.ckpt checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load weights onto (default: cpu)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print module summary after loading",
    )
    args = parser.parse_args()

    module = load_confidence_weights(args.ckpt_path, args.device)

    if args.summary:
        print_module_summary(module)

    # Freeze all parameters (confidence module should be frozen for quality-graft)
    module.requires_grad_(False)
    module.eval()
    print("\nModule set to eval mode with all gradients disabled (frozen).")

    return module


if __name__ == "__main__":
    main()
