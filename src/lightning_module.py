"""Lightning module for create model and set training stages"""

from typing import List

import pytorch_lightning as pl
import torch

from configs.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.train_utils import load_object


class FirstBreaksModule(pl.LightningModule):
    """
    Lightning module for torch model training
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

        self._model = load_object(self._config.train_config.model_type)(
            encoder_name=self._config.train_config.encoder_name,
            encoder_weights=self._config.train_config.encoder_weights,
            classes=config.data_config.n_classes,  # For binary segmentation
            activation=None,  # No activation, as we'll apply sigmoid in the training step
        )
        self._losses = get_losses(self._config.seg_losses)
        self._valid_metrics = get_metrics()
        self._test_metrics = get_metrics()
        self.save_hyperparameters(self._config.dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self) -> dict:
        optimizer = load_object(self._config.train_config.optimizer)(
            self._model.parameters(),
            **self._config.train_config.optimizer_kwargs,
        )
        scheduler = load_object(
            self._config.train_config.scheduler,
        )(optimizer, **self._config.train_config.scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)
        loss = self._calculate_loss(pred_masks_logits, gt_masks)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)
        loss = self._calculate_loss(pred_masks_logits, gt_masks)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._valid_metrics.update(pred_masks, gt_masks)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        images, gt_masks = batch
        gt_masks = gt_masks.long().unsqueeze(1)
        pred_masks_logits = self(images)
        pred_masks = torch.sigmoid(pred_masks_logits)
        self._test_metrics.update(pred_masks, gt_masks)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = self._valid_metrics.compute()
        for key, value in metrics.items():
            self.log(f"val_{key}", value, on_epoch=True, prog_bar=True)
        self._valid_metrics.reset()

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.compute()
        for key, value in metrics.items():
            self.log(f"test_{key}", value, on_epoch=True)
        self._test_metrics.reset()

    def _calculate_loss(
        self,
        pred_masks_logits: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = 0
        for loss_wrapper in self._losses:
            loss = loss_wrapper.loss(pred_masks_logits, gt_masks)
            total_loss += loss_wrapper.weight * loss
        return total_loss
