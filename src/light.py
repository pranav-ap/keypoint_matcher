import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, ModelSummary
from neptune.types import File
from torch.distributions.multivariate_normal import MultivariateNormal

from config import config
from utils import show_batch, get_tensor_grid, logger
from .model import MatcherModel

torch.set_float32_matmul_precision('medium')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = MatcherModel().to(device)

        self.learning_rate = config.train.learning_rate
        self.mae = torchmetrics.MeanAbsoluteError()

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, reference_patches, target_patches, reference_coords, estimates):
        pred = self.model(
            reference_patches,
            target_patches,
            reference_coords,
            estimates,
        )

        return pred

    @staticmethod
    def _compute_coords_accuracy_percentage(coords_pred, target, pixels=1):
        difference = torch.abs(coords_pred - target)

        # Check which differences exceed N pixels
        exceeds = (difference > pixels).any(dim=1)
        # Calculate the percentage
        percentage = (exceeds.sum().item() / coords_pred.shape[0]) * 100

        return percentage

    def _compute_coords_loss(self, coords_pred_norm, references, targets, estimates):
        targets_norm = Light._normalize_coords(targets.float())
        # estimates_norm = Light._normalize_coords(estimates.float())

        loss = F.mse_loss(coords_pred_norm, targets_norm)

        return loss

    def _shared_step(self, batch):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch

        coords_pred_norm, conf_pred = self.model(
            ref_patches,
            tar_patches,
            references,
            estimates,
        )

        coords_pred = Light._denormalize_coords(coords_pred_norm.float())

        coords_loss = self._compute_coords_loss(
            coords_pred_norm, 
            references,
            targets, 
            estimates,
        )

        conf_loss = 0
    
        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=3)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1)

        loss = coords_loss

        return loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel

    def training_step(self, batch, batch_idx):
        _, _, _, targets, _, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        metrics = {
            "train/loss": loss,
            "train/coords_loss": coords_loss,
            "train/confidence_loss": conf_loss,
            "train/coords_mae": self.mae(coords_pred, targets.squeeze(1)),         
            "train/coords_percent_1_pixel": coords_percent_1_pixel,
            "train/coords_percent_1.5_pixel": coords_percent_15_pixel,
            "train/coords_percent_2_pixel": coords_percent_2_pixel,
            "train/coords_percent_3_pixel": coords_percent_3_pixel,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='train')

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, _, _, targets, _, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        metrics = {
            "val/loss": loss,
            "val/coords_loss": coords_loss,
            "val/confidence_loss": conf_loss,
            "val/coords_mae": self.mae(coords_pred, targets.squeeze(1)),         
            "val/coords_percent_1_pixel": coords_percent_1_pixel,
            "val/coords_percent_1.5_pixel": coords_percent_15_pixel,
            "val/coords_percent_2_pixel": coords_percent_2_pixel,
            "val/coords_percent_3_pixel": coords_percent_3_pixel,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='val')

        return metrics

    def _log_images(self, batch, target_coords, rotation_pred=None, confidence_pred=None, limit_count=None, stage=None):
        ref_patches, references, tar_patches, targets, confidences, cert, _ = batch
        
        image_grid = show_batch(
            ref_patches, tar_patches,
            references, 

            patch_level_target_coords=target_coords, patch_level_target_coords_true=targets,

            confidences_true=confidences,
            confidence_pred=confidence_pred,

            limit_count=limit_count,
            n_columns=8,
        )

        name = f'{stage}_epoch_{self.current_epoch}.png'

        if self.neptune_logger is not None:
            self.neptune_logger.experiment[f"{stage}/images"].append(
                File.as_image(image_grid),
                step=self.global_step,
                name=name,
            )

    def configure_optimizers(self):        
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
        )

        lr_scheduler = {
            "scheduler": scheduler,
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

        summary_callback = ModelSummary(max_depth=1)

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback,
            summary_callback,
        ]

    @staticmethod
    def _normalize_coords(coords):
        """Normalizes coordinates from [0, 81] to [-1, 1] in a numerically stable way."""
        return (coords - (config.image.train_patch_size - 1) / 2) / ((config.image.train_patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 81]."""
        return coords * ((config.image.train_patch_size - 1) / 2) + (config.image.train_patch_size - 1) / 2
