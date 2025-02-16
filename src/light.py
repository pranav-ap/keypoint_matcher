import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, ModelSummary
from neptune.types import File

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
        # self.r2 = torchmetrics.R2Score()

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
    def _compute_coords_accuracy_percentage(pred, target, pixels=1):
        # Scale pred from [-1, 1] to [0, 31]
        scaled_pred = (pred.float() + 1) * ((config.image.train_patch_size - 1) / 2) # 15.5 
        
        difference = torch.abs(scaled_pred - target)

        # Check which differences exceed N pixels
        exceeds = (difference > pixels).any(dim=1)
        # Calculate the percentage
        percentage = (exceeds.sum().item() / scaled_pred.shape[0]) * 100

        return percentage

    @staticmethod
    def _compute_coords_loss(pred, target):
        # Scale targets to [-1, 1]
        scaled_target = target.float() / ((config.image.train_patch_size - 1) / 2) - 1  
        
        # MSE Loss
        loss = F.mse_loss(pred, scaled_target)

        # L1 Loss
        # loss = F.l1_loss(pred, scaled_target)

        return loss

    @staticmethod
    def _compute_confidence_loss(logits_pred, confidences, cert):
        # loss = F.binary_cross_entropy_with_logits(logits_pred, cert)
        # loss = F.binary_cross_entropy_with_logits(logits_pred.squeeze(1), confidences)
        loss = F.mse_loss(logits_pred.squeeze(1), confidences)
        return loss

    def _shared_step(self, batch):
        ref_patches, ref_keypoints, tar_patches, tar_keypoints, confidences, cert = batch

        coords_pred, conf_pred = self.model(
            ref_patches,
            tar_patches,
            ref_keypoints,
        )
        
        # Calc Loss

        coords_loss = self._compute_coords_loss(coords_pred, tar_keypoints)
        conf_loss = self._compute_confidence_loss(conf_pred, confidences, cert)

        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=3)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=2)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=1.5)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, tar_keypoints, pixels=1)

        loss = coords_loss * 0.2 + conf_loss * 0.8

        return loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel
        # return loss, coords_loss, coords_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel

    def training_step(self, batch, batch_idx):
        _, _, _, tar_keypoints, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)
        # loss, coords_loss, coords_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        # Log metrics
        coords_pred = (coords_pred + 1) * ((config.image.train_patch_size - 1) / 2) # 15.5

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
            # self._log_images(batch, coords_pred, rotation_pred=rotation_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='train')
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='train')
            # self._log_images(batch, coords_pred, limit_count=limit_count, stage='train')

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, _, _, tar_keypoints, _, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)
        # loss, coords_loss, coords_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        # Log metrics

        coords_pred = (coords_pred + 1) * ((config.image.train_patch_size - 1) / 2) # 15.5

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
            # self._log_images(batch, coords_pred, rotation_pred=rotation_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='val')
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='val')
            # self._log_images(batch, coords_pred, limit_count=limit_count, stage='val')

        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, _, _, tar_keypoints, _ = batch
        loss, coords_loss, coords_pred, conf_loss, conf_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)
        # loss, coords_loss, coords_pred, coords_percent_3_pixel, coords_percent_2_pixel, coords_percent_15_pixel, coords_percent_1_pixel = self._shared_step(batch)

        # Log metrics

        coords_pred = (coords_pred + 1) * ((config.image.train_patch_size - 1) / 2) # 63.5 15.5

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
            # self._log_images(batch, coords_pred, rotation_pred=rotation_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='test')
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='test')
            # self._log_images(batch, coords_pred, limit_count=limit_count, stage='test')

        return metrics

    def _log_images(self, batch, target_coords, rotation_pred=None, confidence_pred=None, limit_count=None, stage=None):
        ref_patches, ref_keypoints, tar_patches, tar_keypoints, confidences, cert = batch
        
        image_grid = show_batch(
            ref_patches, tar_patches,
            ref_keypoints, 

            patch_level_target_coords=target_coords, patch_level_target_coords_true=tar_keypoints,

            confidences_true=confidences,
            confidence_pred=confidence_pred,

            limit_count=limit_count,
            n_columns=8,
        )

        # out_path = None

        # match stage:
        #     case 'train':
        #         out_path = config.paths.output.train_images
        #     case 'val':
        #         out_path = config.paths.output.val_images
        #     case 'test':
        #         out_path = config.paths.output.test_images

        name = f'{stage}_epoch_{self.current_epoch}.png'
        # out_path = os.path.join(out_path, name)
        # image_grid.save(out_path)

        if self.neptune_logger is not None:
            self.neptune_logger.experiment[f"{stage}/images"].append(
                File.as_image(image_grid),
                step=self.global_step,
                name=name,
            )

        # if self.tensorboard_logger is not None:
        #     self.tensorboard_logger.experiment.add_images(
        #         tag=f"{stage}_images",
        #         img_tensor=get_tensor_grid(image_grid),
        #         global_step=self.global_step
        #     )

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
