from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from cellseg_challenge.data import CellDataModule
from cellseg_challenge.model import CellSegmentationModule
from cellseg_challenge.utils import as_plain_dict, rle_encode, seed_everything, write_submission


def train_from_config(cfg: Any) -> str:
    cfg = as_plain_dict(cfg)
    seed_everything(int(cfg.get("seed", 42)))

    data_dir = Path(to_absolute_path(cfg["paths"]["data_dir"]))
    output_dir = Path(to_absolute_path(cfg["paths"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = CellDataModule(
        data_dir=data_dir,
        cfg=cfg["data"],
        aug_cfg=cfg["augmentations"],
        loss_cfg=cfg["loss"],
        seed=int(cfg.get("seed", 42)),
    )
    datamodule.setup("fit")

    model = CellSegmentationModule(
        model_cfg=cfg["model"],
        loss_cfg=cfg["loss"],
        optimizer_cfg=cfg["optimizer"],
        scheduler_cfg=cfg["scheduler"],
        class_weights=datamodule.class_weights,
        max_epochs=int(cfg["trainer"]["max_epochs"]),
    )

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch{epoch:02d}",
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [checkpoint, LearningRateMonitor(logging_interval="epoch")]
    patience = int(cfg["trainer"].get("early_stopping_patience", 0) or 0)
    if patience > 0:
        callbacks.append(EarlyStopping(monitor="val/dice", mode="max", patience=patience))

    trainer = pl.Trainer(
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        accelerator=cfg["trainer"].get("accelerator", "auto"),
        devices=cfg["trainer"].get("devices", "auto"),
        precision=cfg["trainer"].get("precision", "32-true"),
        gradient_clip_val=float(cfg["trainer"].get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(cfg["trainer"].get("log_every_n_steps", 25)),
        deterministic=bool(cfg["trainer"].get("deterministic", False)),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
    )
    trainer.fit(model, datamodule=datamodule)

    return checkpoint.best_model_path


def predict_from_config(cfg: Any, ckpt_path: str | None = None) -> Path:
    cfg = as_plain_dict(cfg)
    data_dir = Path(to_absolute_path(cfg["paths"]["data_dir"]))
    output_path = Path(to_absolute_path(cfg["paths"]["submission_path"]))
    ckpt = ckpt_path or cfg.get("predict", {}).get("ckpt_path")
    if not ckpt:
        raise ValueError("Set predict.ckpt_path or pass ckpt_path to predict_from_config")

    datamodule = CellDataModule(
        data_dir=data_dir,
        cfg=cfg["data"],
        aug_cfg=cfg["augmentations"],
        loss_cfg=cfg["loss"],
        seed=int(cfg.get("seed", 42)),
    )
    datamodule.setup("predict")

    model_cfg = dict(cfg["model"])
    model_cfg["encoder_weights"] = None
    model = CellSegmentationModule.load_from_checkpoint(ckpt, model_cfg=model_cfg)
    trainer = pl.Trainer(
        accelerator=cfg["trainer"].get("accelerator", "auto"),
        devices=cfg["trainer"].get("devices", "auto"),
        precision=cfg["trainer"].get("precision", "32-true"),
        logger=False,
    )
    outputs = trainer.predict(model, datamodule=datamodule)

    threshold = float(cfg["predict"].get("threshold", 0.5))
    ids: list[str] = []
    rles: list[str] = []
    for batch in outputs:
        probs: torch.Tensor = batch["probs"]
        masks = (probs.numpy() >= threshold).astype(np.uint8)
        ids.extend([str(x) for x in batch["ids"]])
        rles.extend(rle_encode(mask) for mask in masks)

    write_submission(output_path, ids, rles, datamodule.predict_mask_column)
    return output_path
