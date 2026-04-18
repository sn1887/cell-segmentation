from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rle_encode(mask: np.ndarray) -> str:
    """Run-length encode a binary mask with row-major, one-indexed pixels."""

    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(x) for x in changes)


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not rle:
        return mask.reshape(shape, order="C")

    values = np.asarray([int(x) for x in rle.split()], dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    for start, end in zip(starts, ends, strict=False):
        mask[start:end] = 1
    return mask.reshape(shape, order="C")


def read_submission_ids(path: Path) -> tuple[list[str], str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ids = [row[0] for row in reader]
    mask_column = header[1] if len(header) > 1 else "Mask"
    return ids, mask_column


def write_submission(path: Path, ids: list[str], rles: list[str], mask_column: str = "Mask") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", mask_column])
        writer.writerows(zip(ids, rles, strict=True))


def resolve_image_path(
    image_dirs: list[Path],
    image_id: str,
    id_width: int = 5,
    suffix: str = ".jpg",
) -> Path:
    candidates = []
    stripped = image_id.strip()
    candidates.append(stripped)
    if stripped.isdigit():
        candidates.append(f"{int(stripped):0{id_width}d}")

    seen = set()
    for directory in image_dirs:
        for stem in candidates:
            path = directory / f"{stem}{suffix}"
            if path in seen:
                continue
            if path.exists():
                return path
            seen.add(path)

    search_space = ", ".join(str(p) for p in image_dirs)
    raise FileNotFoundError(f"Could not resolve image ID {image_id!r} in {search_space}")


def as_plain_dict(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value
