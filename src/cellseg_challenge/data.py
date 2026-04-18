from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cellseg_challenge.augmentations import build_transforms
from cellseg_challenge.losses import mask_to_one_hot, one_hot_to_signed_distance
from cellseg_challenge.utils import read_submission_ids, resolve_image_path


@dataclass(frozen=True)
class TrainRecord:
    image_id: str
    fg_ratio: float


class CellSegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[TrainRecord],
        image_dir: Path,
        mask_dir: Path,
        transforms: Any,
        num_classes: int = 2,
        compute_dist_map: bool = True,
        normalize_dist: bool = True,
    ) -> None:
        self.records = records
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.num_classes = num_classes
        self.compute_dist_map = compute_dist_map
        self.normalize_dist = normalize_dist

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_path = self.image_dir / f"{record.image_id}.jpg"
        mask_path = self.mask_dir / f"{record.image_id}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 0).astype(np.uint8)

        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].astype(np.int64)

        sample: dict[str, Any] = {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask).long(),
            "id": record.image_id,
        }

        if self.compute_dist_map:
            one_hot = mask_to_one_hot(mask, self.num_classes)
            dist_map = one_hot_to_signed_distance(one_hot, normalize=self.normalize_dist)
            sample["dist_map"] = torch.from_numpy(dist_map).float()

        return sample


class InferenceDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        image_dirs: list[Path],
        transforms: Any,
        id_width: int = 5,
    ) -> None:
        self.ids = ids
        self.image_dirs = image_dirs
        self.transforms = transforms
        self.id_width = id_width

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.ids[idx]
        image_path = resolve_image_path(self.image_dirs, image_id, id_width=self.id_width)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=image)
        image = augmented["image"]
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "id": image_id,
            "image_path": str(image_path),
        }


class CellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        cfg: dict[str, Any],
        aug_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.aug_cfg = aug_cfg
        self.loss_cfg = loss_cfg
        self.seed = seed
        self.train_dataset: CellSegmentationDataset | None = None
        self.val_dataset: CellSegmentationDataset | None = None
        self.predict_dataset: InferenceDataset | None = None
        self.class_weights: list[float] | None = None
        self.predict_ids: list[str] = []
        self.predict_mask_column = "Mask"

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            records = self._read_train_records()
            train_records, val_records = self._split_records(records)
            self.class_weights = self._compute_class_weights(records)

            mean = self.aug_cfg["normalize"]["mean"]
            std = self.aug_cfg["normalize"]["std"]
            image_size = int(self.cfg["image_size"])
            train_tfms = build_transforms(
                "train", image_size, mean, std, preset=self.aug_cfg.get("train", "cell")
            )
            val_tfms = build_transforms(
                "val", image_size, mean, std, preset=self.aug_cfg.get("val", "basic")
            )

            image_dir = self.data_dir / self.cfg["train_images"]
            mask_dir = self.data_dir / self.cfg["train_masks"]
            compute_dist_map = bool(self.cfg.get("compute_dist_map", True))
            normalize_dist = bool(self.loss_cfg.get("normalize_dist", True))
            self.train_dataset = CellSegmentationDataset(
                train_records,
                image_dir,
                mask_dir,
                train_tfms,
                num_classes=2,
                compute_dist_map=compute_dist_map,
                normalize_dist=normalize_dist,
            )
            self.val_dataset = CellSegmentationDataset(
                val_records,
                image_dir,
                mask_dir,
                val_tfms,
                num_classes=2,
                compute_dist_map=compute_dist_map,
                normalize_dist=normalize_dist,
            )

        if stage in (None, "predict"):
            mean = self.aug_cfg["normalize"]["mean"]
            std = self.aug_cfg["normalize"]["std"]
            transforms = build_transforms(
                "predict", int(self.cfg["image_size"]), mean, std, preset="basic"
            )
            sample_path = self.data_dir / self.cfg["sample_submission"]
            self.predict_ids, self.predict_mask_column = read_submission_ids(sample_path)
            image_dirs = [self.data_dir / image_dir for image_dir in self.cfg["test_images"]]
            self.predict_dataset = InferenceDataset(
                self.predict_ids,
                image_dirs,
                transforms,
                id_width=int(self.cfg.get("id_width", 5)),
            )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset, shuffle=False)

    def _loader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders")

        num_workers = int(self.cfg.get("num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(self.cfg.get("pin_memory", True)),
            persistent_workers=bool(self.cfg.get("persistent_workers", False)) and num_workers > 0,
        )

    def _read_train_records(self) -> list[TrainRecord]:
        metadata_path = self.data_dir / self.cfg["train_metadata"]
        if metadata_path.exists():
            with metadata_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                return [
                    TrainRecord(row["ID"], float(row.get("fg_ratio", 0.0) or 0.0))
                    for row in reader
                ]

        image_dir = self.data_dir / self.cfg["train_images"]
        return [TrainRecord(path.stem, 0.0) for path in sorted(image_dir.glob("*.jpg"))]

    def _split_records(self, records: list[TrainRecord]) -> tuple[list[TrainRecord], list[TrainRecord]]:
        val_fraction = float(self.cfg.get("val_fraction", 0.15))
        bins = int(self.cfg.get("stratify_bins", 8))
        rng = np.random.default_rng(self.seed)
        ratios = np.asarray([record.fg_ratio for record in records])

        if bins <= 1 or len(np.unique(ratios)) <= 1:
            indices = np.arange(len(records))
            rng.shuffle(indices)
            val_size = max(1, int(round(len(records) * val_fraction)))
            val_indices = set(indices[:val_size])
        else:
            quantiles = np.quantile(ratios, np.linspace(0, 1, bins + 1)[1:-1])
            groups = np.digitize(ratios, quantiles, right=True)
            val_indices = set()
            for group_id in np.unique(groups):
                group_indices = np.flatnonzero(groups == group_id)
                rng.shuffle(group_indices)
                val_size = max(1, int(round(len(group_indices) * val_fraction)))
                val_indices.update(group_indices[:val_size].tolist())

        train_records = [record for idx, record in enumerate(records) if idx not in val_indices]
        val_records = [record for idx, record in enumerate(records) if idx in val_indices]
        return train_records, val_records

    def _compute_class_weights(self, records: list[TrainRecord]) -> list[float] | None:
        setting = self.loss_cfg.get("class_weights")
        if setting in (None, "none", "null", False):
            return None
        if setting != "auto":
            return [float(x) for x in setting]

        fg = float(np.mean([record.fg_ratio for record in records]))
        fg = min(max(fg, 1e-6), 1.0 - 1e-6)
        bg = 1.0 - fg
        return [1.0 / (2.0 * bg), 1.0 / (2.0 * fg)]

