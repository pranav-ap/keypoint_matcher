import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
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

        self.model = MatcherModel()
        self.model = self.model.to('cuda')

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

    def forward(self, reference_patches, target_patches, reference_coords):
        pred = self.model(
            reference_patches,
            target_patches,
            reference_coords,
        )

        return pred

    @staticmethod
    def _compute_coords_loss(pred, target):
        # Scale targets to [-1, 1]
        scaled_target = target.float() / 15.5 - 1  
        loss = F.mse_loss(pred, scaled_target)
        
        return loss

    @staticmethod
    def _compute_rotation_loss(pred, target):      
        pred = pred.squeeze()

        # Scale targets from radians [-pi, pi] to [-1, 1]
        # scaled_target = target / torch.pi
        # loss = F.mse_loss(pred, scaled_target)
        
        # Scale preds from [-1, 1] to radians [-pi, pi]
        scaled_pred = pred * torch.pi
        loss = F.mse_loss(scaled_pred, target)
        
        return loss

    @staticmethod
    def _compute_confidence_loss(prob_pred, coords_pred, coords):
        std_dev = 6.0
        covariance_matrix = torch.diag(torch.tensor([std_dev ** 2, std_dev ** 2], device=coords.device))

        gaussian = torch.distributions.MultivariateNormal(
            loc=coords, 
            covariance_matrix=covariance_matrix
        )

        X = (coords_pred + 1) * 15.5
        raw_probabilities = torch.exp(gaussian.log_prob(X))
        # raw_probabilities = torch.exp(gaussian.log_prob(coords_pred))

        # Y = coords / 31.0
        # max_probabilities = torch.exp(gaussian.log_prob(Y))
        max_probabilities = torch.exp(gaussian.log_prob(coords))

        # Normalize probabilities to make p = 1 when coords_pred = coords
        normalized_probabilities = raw_probabilities / (max_probabilities + 0.000001)

        loss = F.mse_loss(prob_pred.squeeze(1), normalized_probabilities)
        
        return loss

    def _shared_step(self, batch: Match):
        coords_pred, rotation_pred, prob_pred = self.model(
            batch.reference_patches,
            batch.target_patches,
            batch.patch_level_reference_coords,
        )

        # Calc Loss

        coords = batch.patch_level_target_coords
        coords_loss = self._compute_coords_loss(coords_pred, coords)

        rotation = batch.rotations
        rotation_loss = self._compute_rotation_loss(rotation_pred, rotation)

        prob_loss = self._compute_confidence_loss(prob_pred, coords_pred, coords)

        return coords_loss, rotation_loss, prob_loss, coords_pred, rotation_pred, prob_pred

    def training_step(self, batch: Match, batch_idx):
        coords_loss, rotation_loss, confidence_loss, coords_pred, rotation_pred, confidence_pred = self._shared_step(batch)

        # Calc Loss

        loss = coords_loss + rotation_loss + confidence_loss

        # Log metrics

        coords_pred = (coords_pred + 1) * 15.5
        coords = batch.patch_level_target_coords

        metrics = {
            "train/loss": loss,
            "train/coords_loss": coords_loss,
            "train/rotation_loss": rotation_loss,
            "train/confidence_loss": confidence_loss,
            "train/coords_mae": self.mae(coords_pred, coords.squeeze(1)),
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, rotation_pred, confidence_pred, limit_count=limit_count, stage='train')

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Match, batch_idx):
        coords_loss, rotation_loss, confidence_loss, coords_pred, rotation_pred, confidence_pred = self._shared_step(batch)

        # Calc Loss

        loss = coords_loss + rotation_loss + confidence_loss

        # Log metrics

        coords_pred = (coords_pred + 1) * 15.5
        coords = batch.patch_level_target_coords

        metrics = {
            "val/loss": loss,
            "val/coords_loss": coords_loss,
            "val/rotation_loss": rotation_loss,
            "val/confidence_loss": confidence_loss,
            "val/coords_mae": self.mae(coords_pred, coords.squeeze(1)),
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, rotation_pred, confidence_pred, limit_count=limit_count, stage='val')

        return metrics

    @torch.no_grad()
    def test_step(self, batch: Match, batch_idx):
        coords_loss, rotation_loss, confidence_loss, coords_pred, rotation_pred, confidence_pred = self._shared_step(batch)

        # Calc Loss

        loss = coords_loss + rotation_loss + confidence_loss

        # Log metrics

        coords_pred = (coords_pred + 1) * 15.5
        coords = batch.patch_level_target_coords

        metrics = {
            "test/loss": loss,
            "test/coords_loss": coords_loss,
            "test/rotation_loss": rotation_loss,
            "test/confidence_loss": confidence_loss,
            "test/coords_mae": self.mae(coords_pred, coords.squeeze(1)),
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, rotation_pred, confidence_pred, limit_count=limit_count, stage='test')

        return metrics

    def _log_images(self, batch, target_coords, rotation_pred, confidence_pred, limit_count=None, stage=None):
        image_grid = show_batch(
            batch.reference_patches, batch.target_patches,
            batch.patch_level_reference_coords, 
            target_coords, batch.patch_level_target_coords,
            rotations_true=batch.rotations,
            rotations=rotation_pred,
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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

        progress_bar_callback = TQDMProgressBar(refresh_rate=2)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback
        ]
