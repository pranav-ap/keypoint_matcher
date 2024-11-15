import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from config import config
from utils import make_clear_directory
from .model import DescriptorModel

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = DescriptorModel()

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

        make_clear_directory(config.dirs.test_images)
        make_clear_directory(config.dirs.val_images)

    def forward(self, patches):
        return self.model(patches)

    @staticmethod
    def compute_loss(reference_embeddings, target_embeddings):
        # Calculate a loss across all pixels in the patch
        patch_loss = F.mse_loss(reference_embeddings, target_embeddings)
        return patch_loss

    def training_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)

        loss = self.compute_loss(reference_embeddings, target_embeddings)
        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)

        labels = [1] * len(left_coords)

        loss = self.compute_loss(reference_embeddings, target_embeddings, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)

        labels = [1] * len(left_coords)

        loss = self.compute_loss(reference_embeddings, target_embeddings, labels)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.paths.output}/checkpoints/',
            filename="best-checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=10)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]
