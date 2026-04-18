from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def mask_to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((num_classes, *mask.shape), dtype=np.uint8)
    for class_idx in range(num_classes):
        one_hot[class_idx] = mask == class_idx
    return one_hot


def one_hot_to_signed_distance(
    one_hot: np.ndarray,
    resolution: Sequence[float] | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Convert one-hot masks to signed distance maps.

    Positive values are outside a class region and negative values are inside it, matching
    the boundary-loss formulation from LIVIAETS/boundary-loss.
    """

    distances = np.zeros_like(one_hot, dtype=np.float32)
    sampling = resolution if resolution is not None else [1.0] * (one_hot.ndim - 1)

    for class_idx in range(one_hot.shape[0]):
        posmask = one_hot[class_idx].astype(bool)
        if not posmask.any():
            continue
        negmask = ~posmask
        outside = distance_transform_edt(negmask, sampling=sampling) * negmask
        inside = (distance_transform_edt(posmask, sampling=sampling) - 1.0) * posmask
        distances[class_idx] = outside - inside

        if normalize:
            scale = np.abs(distances[class_idx]).max()
            if scale > 0:
                distances[class_idx] /= scale

    return distances


class BoundaryLoss(nn.Module):
    """Boundary loss over pre-computed signed distance maps."""

    def __init__(self, idc: Sequence[int] = (1,)) -> None:
        super().__init__()
        self.idc = list(idc)

    def forward(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs = probs[:, self.idc, ...]
        dist_maps = dist_maps[:, self.idc, ...]
        return torch.einsum("bcwh,bcwh->bcwh", probs, dist_maps).mean()


class CrossEntropyBoundaryLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        boundary_weight: float = 0.01,
        boundary_idc: Sequence[int] = (1,),
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.boundary = BoundaryLoss(boundary_idc)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        dist_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        total = self.ce_weight * ce
        parts = {"ce": ce.detach()}

        if self.boundary_weight and dist_maps is not None:
            probs = torch.softmax(logits, dim=1)
            boundary = self.boundary(probs, dist_maps.float())
            total = total + self.boundary_weight * boundary
            parts["boundary"] = boundary.detach()

        parts["total"] = total.detach()
        return total, parts

