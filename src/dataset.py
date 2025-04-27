"""Module for Dataset preparation"""

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as albu
import jpeg4py as jpeg
import cv2
import numpy as np
from torch.utils.data import Dataset


class FirstBreaksDataset(Dataset):
    """
    Dataset module for torch
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transforms: Optional[albu.BaseCompose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.array]:
        img_path = Path(self.image_paths[idx])
        image = jpeg.JPEG(img_path).decode()
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transforms is not None:
            out = self.transforms(
                image=image,
                mask=mask,
            )
            image = out["image"]
            mask = out["mask"]
        return image, mask
