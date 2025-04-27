"""Module for self-wrote log functions"""

import numpy as np
from pytorch_lightning import Callback, Trainer
from torchvision.utils import make_grid
from .. import augmentations

from src.lightning_module import FirstBreaksModule


class VisualizePredMask(Callback):
    """Visualize mask predictions"""

    def __init__(
            self,
            every_n_epochs: int,
            batch_num: int = 8,
            image_width: int = 512,
            image_height: int = 512,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.batch_num = batch_num
        self.image_width = image_width
        self.image_height = image_height

    def on_train_epoch_start(self, trainer: Trainer, pl_module: FirstBreaksModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        images, gt_masks = next(iter(trainer.train_dataloader))

        visualizations = []
        for img in images:
            img = augmentations.denormalize(augmentations.tensor_to_cv_image(img))
            visualizations.append(augmentations.cv_image_to_tensor(img, normalize=False))

        for gt_mask in gt_masks:
            gt_mask = gt_mask.squeeze().cpu().detach().numpy()
            green_masks = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            valid_area = np.argwhere(gt_mask > 0)
            green_masks[valid_area[:, 0], valid_area[:, 1], 1] = 255
            visualizations.append(augmentations.cv_image_to_tensor(green_masks, normalize=False))

        logits = pl_module(images.to(device=pl_module.device))
        for logit_i, logit in enumerate(logits):
            logit = (logit.squeeze().cpu().detach().numpy())
            pr_mask = np.exp(-np.logaddexp(0, -logit))  # sigmoid
            green_masks = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            valid_area = np.argwhere(pr_mask > 0.5)
            green_masks[valid_area[:, 0], valid_area[:, 1], 1] = 255
            visualizations.append(augmentations.cv_image_to_tensor(green_masks, normalize=False))

        grid = make_grid(visualizations, nrow=self.batch_num, normalize=False)
        trainer.logger.experiment.add_image(
            'Mask preview',
            img_tensor=grid,
            global_step=trainer.global_step,
        )
