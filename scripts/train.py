#!/usr/bin/env python
"""Quality-Graft training script.

Usage:
    # Preprocess only (downloads, PyG conversion, Boltz-1 pLDDT labels)
    python scripts/train.py --mode=preprocess

    # Train (assumes preprocessing is done)
    python scripts/train.py --mode=train

    # Override config values
    python scripts/train.py --mode=train training.max_epochs=10 training.batch_size=2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# Ensure project paths are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LA_PROTEINA_DIR = SRC_DIR / "la_proteina"
for p in [PROJECT_ROOT, SRC_DIR, LA_PROTEINA_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from la_proteina.proteinfoundation.datasets.pdb_data import (
    PDBDataSelector,
    PDBDataSplitter,
)
from quality_graft.data.datamodule import QualityGraftDataModule
from quality_graft.models.adaptor import AdaptorModule
from quality_graft.models.confidence_head import BoltzConfidenceHead
from quality_graft.models.la_proteina_wrapper import LaProteinaWrapper
from quality_graft.models.quality_graft import QualityGraft
from quality_graft.training.lightning_module import QualityGraftLightningModule

logger = logging.getLogger(__name__)


def build_data_module(cfg: DictConfig) -> QualityGraftDataModule:
    """Build the data module from Hydra config."""
    data_cfg = cfg.data.dataset

    dataselector = PDBDataSelector(
        data_dir=data_cfg.data_dir,
        max_length=data_cfg.max_length,
        min_length=data_cfg.min_length,
        molecule_type=data_cfg.molecule_type,
        oligomeric_min=data_cfg.oligomeric_min,
        oligomeric_max=data_cfg.oligomeric_max,
    )
    datasplitter = PDBDataSplitter(data_dir=data_cfg.data_dir)

    boltz_config = OmegaConf.to_container(data_cfg.boltz, resolve=True)

    return QualityGraftDataModule(
        data_dir=data_cfg.data_dir,
        dataselector=dataselector,
        datasplitter=datasplitter,
        format=data_cfg.format,
        boltz_config=boltz_config,
        num_plddt_bins=data_cfg.num_plddt_bins,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )


def build_model(cfg: DictConfig) -> QualityGraft:
    """Build the full QualityGraft model from Hydra config."""
    model_cfg = cfg.model

    # La-Proteina wrapper (from checkpoint)
    lp_cfg = model_cfg.la_proteina_wrapper
    la_proteina = LaProteinaWrapper.from_checkpoint(
        proteina_ckpt_path=lp_cfg.proteina_ckpt_path,
        autoencoder_ckpt_path=lp_cfg.autoencoder_ckpt_path,
        device=lp_cfg.device,
        use_decoder=lp_cfg.use_decoder,
        t_value=lp_cfg.t_value,
        deterministic_encode=lp_cfg.deterministic_encode,
    )

    # Adaptor (via Hydra instantiate)
    adaptor = hydra.utils.instantiate(model_cfg.quality_graft.adaptor)

    # Confidence head
    ch_cfg = model_cfg.quality_graft.confidence_head
    confidence_head = BoltzConfidenceHead(
        token_s=ch_cfg.token_s,
        token_z=ch_cfg.token_z,
        pairformer_args=OmegaConf.to_container(ch_cfg.pairformer_args, resolve=True),
        confidence_model_args=OmegaConf.to_container(ch_cfg.confidence_model_args, resolve=True),
        full_embedder_args=OmegaConf.to_container(ch_cfg.full_embedder_args, resolve=True),
        msa_args=OmegaConf.to_container(ch_cfg.msa_args, resolve=True),
        ckpt_path=ch_cfg.ckpt_path,
        ckpt_prefix=ch_cfg.ckpt_prefix,
        device=ch_cfg.device,
        freeze=ch_cfg.freeze,
        strict_loading=ch_cfg.strict_loading,
    )

    return QualityGraft(
        la_proteina=la_proteina,
        adaptor=adaptor,
        confidence_head=confidence_head,
    )


def build_lightning_module(cfg: DictConfig, model: QualityGraft) -> QualityGraftLightningModule:
    """Wrap the model in a Lightning module."""
    train_cfg = cfg.training

    return QualityGraftLightningModule(
        model=model,
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
        warmup_steps=train_cfg.scheduler.warmup_steps,
        min_lr=train_cfg.scheduler.min_lr,
        num_plddt_bins=cfg.data.dataset.num_plddt_bins,
    )


def build_trainer(cfg: DictConfig) -> L.Trainer:
    """Build the Lightning Trainer."""
    train_cfg = cfg.training

    # W&B logger
    wandb_logger = WandbLogger(
        project=train_cfg.wandb.project,
        entity=train_cfg.wandb.entity,
        name=train_cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    return L.Trainer(
        max_epochs=train_cfg.max_epochs,
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
    )


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    # Parse mode from sys.argv (before Hydra consumes args)
    mode = "train"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
            break

    logger.info("Mode: %s", mode)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if mode == "preprocess":
        dm = build_data_module(cfg)
        dm.prepare_data()
        logger.info("Preprocessing complete.")

    elif mode == "train":
        dm = build_data_module(cfg)
        dm.setup("fit")

        model = build_model(cfg)
        lit_module = build_lightning_module(cfg, model)
        trainer = build_trainer(cfg)

        logger.info(
            "Trainable params: %d, Frozen params: %d",
            model.num_trainable_parameters(),
            model.num_frozen_parameters(),
        )

        trainer.fit(lit_module, datamodule=dm)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'preprocess' or 'train'.")


if __name__ == "__main__":
    main()