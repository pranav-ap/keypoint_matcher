import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from neptune.types import File

from config import config
from utils import show_batch, get_tensor_grid
from .dataset import Match
from .model import MatcherModel

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.matcher_model = MatcherModel()

        self.learning_rate = config.train.learning_rate

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        target_coords_pred = self.matcher_model(
            reference_patches,
            target_patches,
            reference_coords,
        )

        return target_coords_pred

    @staticmethod
    def _compute_loss(target_coords, target_coords_pred):
        loss = F.mse_loss(
            target_coords_pred,
            target_coords.float(),
        )

        return loss

    def _shared_step(self, batch: Match):
        target_coords_pred = self.matcher_model(
            batch.reference_patches,
            batch.target_patches,
            batch.patch_level_reference_coords,
        )

        loss = self._compute_loss(
            batch.patch_level_target_coords,
            target_coords_pred
        )

        return loss, target_coords_pred

    def training_step(self, batch: Match, batch_idx):
        target_coords_pred = self.matcher_model(
            batch.reference_patches,
            batch.target_patches,
            batch.patch_level_reference_coords,
        )

        loss = self._compute_loss(
            batch.patch_level_target_coords,
            target_coords_pred
        )

        metrics = {
            "train/loss": loss,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(
                batch,
                target_coords_pred,
                limit_count=limit_count,
                stage='train'
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Match, batch_idx):
        loss, target_coords_pred = self._shared_step(batch)

        metrics = {
            "val/loss": loss,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(
                batch,
                target_coords_pred,
                limit_count=limit_count,
                stage='val'
            )

        return metrics

    @torch.no_grad()
    def test_step(self, batch: Match, batch_idx):
        loss, target_coords_pred = self._shared_step(batch)

        metrics = {
            "test/loss": loss,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(
                batch,
                target_coords_pred,
                limit_count=limit_count,
                stage='test'
            )

        return metrics

    def _log_images(self, batch, target_coords, limit_count=None, stage=None):
        image_grid = show_batch(
            batch.reference_patches, batch.target_patches,
            batch.patch_level_reference_coords, target_coords,
            limit_count=limit_count,
            n_columns=3,
        )

        out_path = config.paths.output.test_images

        if stage == 'train':
            out_path = config.paths.output.train_images
        elif stage == 'val':
            out_path = config.paths.output.val_images 

        name = f'{stage}_global_step_{self.global_step}.png'
        out_path = os.path.join(out_path, name)
        image_grid.save(out_path)

        if self.neptune_logger is not None:
            self.neptune_logger.experiment[f"{stage}/images"].append(
                File.as_image(image_grid),
                step=self.global_step,
                name=name,
            )

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.experiment.add_images(
                tag=f"{stage}_images",
                img_tensor=get_tensor_grid(image_grid),
                global_step=self.global_step
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.matcher_model.parameters(),
            lr=self.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=2,
                factor=0.5
            ),
            "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            dirpath=config.paths.output.checkpoints,
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=5)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback
        ]
