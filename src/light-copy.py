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

        self.register_buffer("coords_weight", torch.tensor(1.0))
        self.register_buffer("conf_weight", torch.tensor(1.0))
        
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
        pred = self.model(
            reference_patches,
            target_patches,
            reference_coords,
        )

        return pred

    @staticmethod
    def _normalize_coords(coords):
        """Normalizes coordinates from [0, 31] to [-1, 1] in a numerically stable way."""
        return (coords - (config.image.train_patch_size - 1) / 2) / ((config.image.train_patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 31]."""
        return coords * ((config.image.train_patch_size - 1) / 2) + (config.image.train_patch_size - 1) / 2

    @staticmethod
    def _compute_coords_accuracy_percentage(pred, target, pixels=1):
        # scaled_pred = Light._denormalize_coords(pred.float())
        scaled_pred = pred

        difference = torch.abs(scaled_pred - target)

        # Check which differences exceed N pixels
        exceeds = (difference > pixels).any(dim=1)
        # Calculate the percentage
        percentage = (exceeds.sum().item() / scaled_pred.shape[0]) * 100

        return percentage

    @staticmethod
    def _compute_coords_loss3(delta_normalized, target, target_est):
        normalized_target = Light._normalize_coords(target.float())
        normalized_target_est = Light._normalize_coords(target_est.float())

        delta_target = normalized_target - normalized_target_est  # True delta
        delta_pred = delta_normalized - normalized_target_est  # Predicted delta

        loss = F.mse_loss(delta_pred, delta_target)  

        # Regularization: Penalize large delta
        # loss_reg = F.mse_loss(delta_pred, torch.zeros_like(delta_pred))  
        # loss = loss + 0.1 * loss_reg  

        return loss

    def _shared_step(self, batch):
        ref_patches, ref_keypoints, tar_patches, tar_keypoints, confidences, cert, tar_est_keypoints = batch

        delta_pred, conf_pred = self.model(
            ref_patches,
            tar_patches,
            ref_keypoints,
        )

        coords_loss = self._compute_coords_loss3(
            delta_pred, 
            tar_keypoints, 
            tar_est_keypoints
        )

        conf_loss = 0

        delta_pred = self._denormalize_coords(delta_pred)
        coords_pred = delta_pred + tar_est_keypoints.float() # denormal 
    
        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=3)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=2)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=1.5)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=1)

        # with torch.no_grad():
        #     loss_ratio = coords_loss / (conf_loss + 1e-6)
        #     self.coords_weight = 1 / (1 + loss_ratio)
        #     self.conf_weight = loss_ratio / (1 + loss_ratio)

        # loss = self.coords_weight * coords_loss + self.conf_weight * conf_loss

        # loss = coords_loss + conf_loss * 0.8
        loss = coords_loss

        coords_pred = torch.clamp(coords_pred, -81, 81)
                
        return loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel

    def training_step(self, batch, batch_idx):
        _, _, _, tar_keypoints, _, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        metrics = {
            "train/loss": loss,
            "train/coords_loss": coords_loss,
            "train/confidence_loss": conf_loss,
            "train/coords_mae": self.mae(coords_pred, tar_keypoints.squeeze(1)),         
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
        _, _, _, tar_keypoints, _, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        metrics = {
            "val/loss": loss,
            "val/coords_loss": coords_loss,
            "val/confidence_loss": conf_loss,
            "val/coords_mae": self.mae(coords_pred, tar_keypoints.squeeze(1)),         
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

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, _, _, tar_keypoints, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        metrics = {
            "test/loss": loss,
            "test/coords_loss": coords_loss,
            "test/confidence_loss": conf_loss,
            "test/coords_mae": self.mae(coords_pred, tar_keypoints.squeeze(1)),         
            "test/coords_percent_1_pixel": coords_percent_1_pixel,
            "test/coords_percent_1.5_pixel": coords_percent_15_pixel,
            "test/coords_percent_2_pixel": coords_percent_2_pixel,
            "test/coords_percent_3_pixel": coords_percent_3_pixel,
        }
        
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='test')

        return metrics

    def _log_images(self, batch, target_coords, rotation_pred=None, confidence_pred=None, limit_count=None, stage=None):
        ref_patches, ref_keypoints, tar_patches, tar_keypoints, confidences, cert, _ = batch
        
        image_grid = show_batch(
            ref_patches, tar_patches,
            ref_keypoints, 

            patch_level_target_coords=target_coords, patch_level_target_coords_true=tar_keypoints,

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
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.5,
        # )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
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

        summary_callback = ModelSummary(max_depth=1)

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback,
            summary_callback,
        ]

    @staticmethod
    def _compute_confidence_loss(pred, confidences):
        # loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), confidences)

        # threshold = 0.05
        # confidences = (confidences > threshold).float()

        # loss = F.binary_cross_entropy(pred.squeeze(1), confidences)

        loss = F.mse_loss(pred.squeeze(1), confidences)
        # loss = F.smooth_l1_loss(pred.squeeze(1), confidences, beta=0.3) 
        # loss = F.smooth_l1_loss(pred.squeeze(1), confidences)
        
        # bar chart over bins of confidence

        return loss

    @staticmethod
    def _compute_confidence_loss2(conf_pred, coords_pred, confidences, tar_keypoints, sigma=2.0):
        loss = F.binary_cross_entropy_with_logits(conf_pred.squeeze(1), confidences, reduction='none')

        # Compute Gaussian weights
        means = tar_keypoints.float()  # Shape: [B, 2]
        cov_matrix = torch.tensor([[sigma**2, 0], [0, sigma**2]], dtype=torch.float32, device=tar_keypoints.device)

        dist = MultivariateNormal(means, covariance_matrix=cov_matrix.expand(means.shape[0], 2, 2))
        peak_prob = dist.log_prob(means).exp()  # Shape: [B]
        scale_factors = confidences / peak_prob

        # Compute weights for predicted coordinates
        weights = dist.log_prob(coords_pred.float()).exp() * scale_factors.unsqueeze(-1)

        weighted_loss = (loss * weights).mean()

        return weighted_loss

    @staticmethod
    def _compute_coords_loss(pred, target):
        normalized_target = Light._normalize_coords(target.float())
        
        loss = F.mse_loss(pred, normalized_target) 
        # loss = F.l1_loss(pred, scaled_target)
        # loss = F.smooth_l1_loss(pred, scaled_target, beta=0.3) 

        return loss

    @staticmethod
    def _compute_coords_loss2(pred, target, true_confidences):
        normalized_target = Light._normalize_coords(target.float())
        
        loss = F.mse_loss(pred, normalized_target, reduction='none')  # Shape: [B, 2]
        weighted_loss = (loss * true_confidences.unsqueeze(-1)).mean()  # Shape: Scalar

        return weighted_loss
        
    