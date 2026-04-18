from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from cellseg_challenge.losses import CrossEntropyBoundaryLoss


class CellSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        class_weights: list[float] | None = None,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg or {
            "arch": "FPN",
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 2,
        }
        self.loss_cfg = loss_cfg or {
            "ce_weight": 1.0,
            "boundary_weight": 0.01,
            "boundary_idc": [1],
        }
        self.optimizer_cfg = optimizer_cfg or {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-5}
        self.scheduler_cfg = scheduler_cfg or {"name": "cosine", "min_lr": 1e-6}
        self.max_epochs = max_epochs
        self.save_hyperparameters()

        self.net = self._build_model(self.model_cfg)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.criterion = CrossEntropyBoundaryLoss(
            ce_weight=float(self.loss_cfg.get("ce_weight", 1.0)),
            boundary_weight=float(self.loss_cfg.get("boundary_weight", 0.0)),
            boundary_idc=self.loss_cfg.get("boundary_idc", [1]),
            class_weights=weight_tensor,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, parts, metrics = self._shared_step(batch)
        self._log_parts("train", parts, metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, parts, metrics = self._shared_step(batch)
        self._log_parts("val", parts, metrics, on_step=False, on_epoch=True)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        logits = self(batch["image"])
        probs = torch.softmax(logits, dim=1)[:, 1]
        return {"ids": batch["id"], "probs": probs.detach().cpu()}

    def configure_optimizers(self) -> Any:
        opt_name = str(self.optimizer_cfg.get("name", "adamw")).lower()
        lr = float(self.optimizer_cfg.get("lr", 3e-4))
        weight_decay = float(self.optimizer_cfg.get("weight_decay", 1e-5))

        if opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=float(self.optimizer_cfg.get("momentum", 0.9)),
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        sched_name = str(self.scheduler_cfg.get("name", "none")).lower()
        if sched_name in ("none", "null", ""):
            return optimizer
        if sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(self.scheduler_cfg.get("factor", 0.5)),
                patience=int(self.scheduler_cfg.get("patience", 3)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/dice"},
            }

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.scheduler_cfg.get("t_max", self.max_epochs)),
            eta_min=float(self.scheduler_cfg.get("min_lr", 1e-6)),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits = self(batch["image"])
        loss, parts = self.criterion(logits, batch["mask"], batch.get("dist_map"))
        metrics = binary_segmentation_metrics(logits, batch["mask"])
        return loss, parts, metrics

    def _log_parts(
        self,
        prefix: str,
        parts: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in parts.items():
            self.log(f"{prefix}/loss_{name}", value, prog_bar=name == "total", on_step=on_step, on_epoch=on_epoch)
        for name, value in metrics.items():
            self.log(f"{prefix}/{name}", value, prog_bar=prefix == "val", on_step=on_step, on_epoch=on_epoch)

    @staticmethod
    def _build_model(cfg: dict[str, Any]) -> nn.Module:
        return smp.create_model(
            cfg.get("arch", "FPN"),
            encoder_name=cfg.get("encoder_name", "resnet34"),
            encoder_weights=cfg.get("encoder_weights"),
            in_channels=int(cfg.get("in_channels", 3)),
            classes=int(cfg.get("classes", 2)),
            activation=None,
        )


def binary_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    pred = torch.argmax(logits, dim=1).bool()
    target = target.bool()
    intersection = (pred & target).sum(dim=(1, 2)).float()
    pred_sum = pred.sum(dim=(1, 2)).float()
    target_sum = target.sum(dim=(1, 2)).float()
    union = (pred | target).sum(dim=(1, 2)).float()
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {"dice": dice, "iou": iou}

