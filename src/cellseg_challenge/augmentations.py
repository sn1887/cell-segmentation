from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
import cv2


def build_transforms(
    split: str,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    preset: str = "basic",
) -> A.Compose:
    resize_normalize = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ]

    if split != "train" or preset == "basic":
        return A.Compose(resize_normalize)

    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.65,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=40,
                        sigma=6,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.2),
                    A.OpticalDistortion(distort_limit=0.12, shift_limit=0.04),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18),
                    A.RandomGamma(gamma_limit=(80, 125)),
                ],
                p=0.55,
            ),
            A.OneOf(
                [
                    A.GaussNoise(),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=3),
                ],
                p=0.25,
            ),
            A.CoarseDropout(p=0.2),
            *resize_normalize,
        ]
    )
