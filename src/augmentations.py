"""Module for set augmentations"""

from typing import Callable, Tuple

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
    preprocessing_fn: Callable = None,
) -> albu.BaseCompose:
    transforms = []
    if preprocessing:
        transforms.append(
            albu.Resize(height=height, width=width),
        )
    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.3),
                albu.CoarseDropout(
                    num_holes_range=(1, 10),
                    hole_height_range=(1, 10),
                    hole_width_range=(1, 10),
                ),
                albu.OneOf(
                    [
                        albu.GaussianBlur(
                            blur_limit=(1, 3),
                            sigma_limit=(0.5, 3),
                        ),
                        albu.ISONoise(
                            color_shift=(0.01, 0.05),
                            intensity=(0.1, 0.3),
                        ),
                        albu.ImageCompression(
                            quality_range=(60, 100),
                        ),
                    ],
                    p=0.5,
                ),
            ],
        )

    if postprocessing:
        if preprocessing_fn is not None:
            transforms.extend(
                [
                    albu.Lambda(image=preprocessing_fn),
                    ToTensorV2(),
                ],
            )
        else:
            transforms.extend(
                [
                    albu.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                    ToTensorV2(),
                ],
            )

    return albu.Compose(
        transforms,
    )


def cv_image_to_tensor(img: NDArray[float], normalize: bool = True) -> Tensor:
    ops = [ToTensorV2()]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)["image"]


def denormalize(
    img: NDArray[float],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
) -> NDArray[int]:
    denorm = albu.Normalize(
        mean=tuple([-me / st for me, st in zip(mean, std)]),  # noqa: WPS221
        std=tuple([1.0 / st for st in std]),
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)["image"] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    return tensor.permute(1, 2, 0).cpu().numpy()
