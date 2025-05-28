import numpy as np 
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar
)
from neptune.types import File
from config import config
from utils import show_batch, logger
from .positional_encoding import RoPENd, positionalencoding2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('medium')


class PreActDSBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.depthwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x

        out = self.depthwise1(out)
        out = self.pointwise1(out)
        out = F.relu(self.bn2(out))
        out = self.depthwise2(out)
        out = self.pointwise2(out)

        out += shortcut
        return out


class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.image.patch_size
        in_channels = 1
        embedding_length = 64
        out_channels = 512

        logger.info(f'{patch_size=}')
        logger.info(f'{in_channels=}')
        logger.info(f'{embedding_length=}')
        logger.info(f'{out_channels=}')

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )
        
        # self.register_buffer(
        #     'positional_encoding', 
        #     positionalencoding2d(embedding_length, height=patch_size, width=patch_size).unsqueeze(0)
        # )

        self.positional_encoding = RoPENd((patch_size, patch_size, embedding_length))

        layers = [
            (embedding_length * 2, 256, 1),
            (256, 512, 1),
            (512, out_channels, 2),
        ]

        self.backbone = nn.ModuleList([
            PreActDSBasicBlock(in_c, out_c, stride) for in_c, out_c, stride in layers
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(out_channels + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
     
        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, estimates):
        _, _, height, width = ref_patches.shape

        # normalize to 0 to 1
        estimates = estimates / (height - 1)
        # normalize to -1 to 1
        estimates = estimates * 2 - 1

        """
        Patch Embedding
        """

        ref_patches = self.patch_embedding(ref_patches)
        
        # ref_patches = ref_patches * self.positional_encoding
        
        # (b, c, h, w) -> (b, h, w, c)
        ref_patches = ref_patches.permute(0, 2, 3, 1).contiguous()
        ref_patches = self.positional_encoding(ref_patches)
        # (b, h, w, c) -> (b, c, h, w)
        ref_patches = ref_patches.permute(0, 3, 1, 2).contiguous()

        tar_patches = self.patch_embedding(tar_patches)
        
        # tar_patches = tar_patches * self.positional_encoding
        
        # (b, c, h, w) -> (b, h, w, c)
        tar_patches = tar_patches.permute(0, 2, 3, 1).contiguous()
        tar_patches = self.positional_encoding(tar_patches)
        # (b, h, w, c) -> (b, c, h, w)
        tar_patches = tar_patches.permute(0, 3, 1, 2).contiguous()

        """
        PreActBasicBlock ResNet 
        """

        x = torch.cat([ref_patches, tar_patches], dim=1)

        for layer in self.backbone:
            x = layer(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        """
        Linear Layers for Separate Heads
        """
        
        x = torch.cat([x, estimates], dim=1)
        x = self.head(x)
        
        coords = F.tanh(x[:, :2]) 
        confs = F.sigmoid(x[:, 2:])  # (B, 2)
        confs_box, confs_point = confs[:, 0], confs[:, 1]  # (B,), (B,)
        
        return coords, confs_box, confs_point

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization for Conv2d layers (for ReLU activation)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier (Glorot) initialization for Linear layers
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)





class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = MatcherModel() 

        self.learning_rate = config.train.learning_rate

        self.mae = torchmetrics.MeanAbsoluteError()
        self.conf_accuracy = torchmetrics.Accuracy(task="binary")
        self.conf_precision = torchmetrics.Precision(task="binary")
        self.conf_recall = torchmetrics.Recall(task="binary")
        self.conf_f1 = torchmetrics.F1Score(task="binary")
        self.conf_matrix = torchmetrics.ConfusionMatrix(task="binary")

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, reference_patches, target_patches, estimates):    
        pred = self.model(
            reference_patches,
            target_patches,
            estimates,
        )

        return pred

    @staticmethod
    def _compute_coords_accuracy_percentage(coords_pred, target, pixels=1.0):
        difference = torch.abs(coords_pred - target)
        exceeds = (difference > pixels).any(dim=1)
        percentage = (exceeds.sum().item() / coords_pred.shape[0]) * 100

        return percentage
    
    @staticmethod
    def _compute_point_confidence_loss(conf_pred, coords_pred, coords_targets):
        std_dev = 2.0
        covariance_matrix = torch.diag(torch.tensor([std_dev ** 2, std_dev ** 2], device=coords_targets.device))

        gaussian = torch.distributions.MultivariateNormal(
            loc=coords_targets, 
            covariance_matrix=covariance_matrix
        )

        raw_probabilities = torch.exp(gaussian.log_prob(coords_pred))
        max_probabilities = torch.exp(gaussian.log_prob(coords_targets))

        # Normalize probabilities to make p = 1 when coords_pred = coords
        normalized_probabilities = raw_probabilities / (max_probabilities + 0.000001)

        loss = F.mse_loss(conf_pred, normalized_probabilities)
        
        return loss
    
    def _shared_step(self, batch, stage='train'):
        ref_patches, references, tar_patches, targets, estimates, confs_true = batch
        
        coords_delta_norm_pred, confs_box_pred, confs_point_pred = self.model(
            ref_patches,
            tar_patches,
            estimates,
        ) 
        
        # Convert coordinate delta to prediction
        coords_delta_pred = coords_delta_norm_pred * ((config.image.patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred

        coords_delta_pred = coords_delta_norm_pred.float() * ((config.image.patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred  

        confs_box_pred_binary = (confs_box_pred > config.image.confidence_decision_threshold).float()
        confs_true_binary = confs_true.float() 
                
        confs_point_loss = self._compute_point_confidence_loss(confs_point_pred, coords_pred, targets)

        # Compute losses
        
        targets_norm = Light._normalize_coords(targets.float())
        estimates_norm = Light._normalize_coords(estimates.float())
        coords_delta_norm_true = targets_norm - estimates_norm

        # coords_loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true)
        # coords_loss = F.smooth_l1_loss(coords_delta_norm_pred, coords_delta_norm_true, beta=0.1)

        # coords_error = coords_delta_norm_pred - coords_delta_norm_true
        # coords_loss = torch.mean(torch.log(1 + 1000 * coords_error.pow(2)))
        
        coords_error = coords_delta_norm_pred - coords_delta_norm_true
        coords_loss = torch.mean(torch.log(1 + 100 * coords_error.pow(2)))
        
        confs_loss = F.binary_cross_entropy(confs_box_pred, confs_true_binary)
       
        # Weighted loss
        
        # alpha = 0.05   
        alpha =  1 / 3 

        # alpha = coords_loss.item() / (confs_loss.item() + 1e-8)
        
        alpha_coords_loss = coords_loss
        alpha_confs_loss = alpha * confs_loss
        alpha_confs_point_loss = alpha * confs_point_loss
        
        loss = alpha_coords_loss + alpha_confs_loss + alpha_confs_point_loss
        
        # Calculate Coords metrics

        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=3.0)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2.0)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.0)
        coords_percent_05_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=0.5)

        coords_mae = self.mae(coords_pred, targets)

        # Calculate Confs metrics

        conf_accuracy = self.conf_accuracy(confs_box_pred_binary, confs_true_binary)

        conf_matrix = self.conf_matrix(confs_box_pred, confs_true_binary)
        tn, fp, fn, tp = conf_matrix.flatten()
        
        precision_0 = tn / (tn + fn + 1e-8)
        recall_0 = tn / (tn + fp + 1e-8)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + 1e-8)
        
        precision_1 = tp / (tp + fp + 1e-8)
        recall_1 = tp / (tp + fn + 1e-8)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)

        # Collect

        # clamp for visualization and further usage
        coords_pred = torch.clamp(coords_pred, min=0.0, max=float(config.image.patch_size))

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/coords_loss": coords_loss,
            f"{stage}/conf_loss": confs_loss, 
            f"{stage}/alpha_coords_loss": alpha_coords_loss, 
            f"{stage}/alpha_confs_loss": alpha_confs_loss, 
            f"{stage}/alpha_confs_point_loss": alpha_confs_point_loss, 

            f"{stage}/coords_mae": coords_mae,
            f"{stage}/coords_percent_3_pixel": coords_percent_3_pixel,
            f"{stage}/coords_percent_2_pixel": coords_percent_2_pixel,
            f"{stage}/coords_percent_15_pixel": coords_percent_15_pixel,
            f"{stage}/coords_percent_125_pixel": coords_percent_125_pixel,
            f"{stage}/coords_percent_1_pixel": coords_percent_1_pixel,
            f"{stage}/coords_percent_05_pixel": coords_percent_05_pixel,
            
            f"{stage}/conf_accuracy": conf_accuracy,

            f"{stage}/conf_precision_0": precision_0.float(),
            f"{stage}/conf_recall_0": recall_0.float(),
            f"{stage}/conf_f1_0": f1_0.float(),
            f"{stage}/conf_precision_1": precision_1.float(),
            f"{stage}/conf_recall_1": recall_1.float(),
            f"{stage}/conf_f1_1": f1_1.float(),

            f"{stage}/conf_tp": tp.float(),
            f"{stage}/conf_fp": fp.float(),
            f"{stage}/conf_fn": fn.float(),
            f"{stage}/conf_tn": tn.float(),
        }

        return metrics, coords_pred, confs_box_pred, confs_point_pred

    @torch.no_grad()
    def _test_step(self, batch, stage='test'):
        ref_patches, references, tar_patches, targets, estimates, confs_true = batch

        coords_delta_norm_pred, confs_pred = self.model(
            ref_patches,
            tar_patches,
            estimates,
        ) 

        coords_delta_pred = coords_delta_norm_pred.float() * ((config.image.patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred  
        
        confs_pred_binary = (confs_pred > config.image.confidence_decision_threshold).float()
        confs_true_binary = confs_true.float() 
        
        # Compute losses
        
        targets_norm = Light._normalize_coords(targets.float())
        estimates_norm = Light._normalize_coords(estimates.float())
        coords_delta_norm_true = targets_norm - estimates_norm

        # coords_loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true)
        # coords_loss = F.smooth_l1_loss(coords_delta_norm_pred, coords_delta_norm_true, beta=0.1)

        coords_error = coords_delta_norm_pred - coords_delta_norm_true
        coords_loss = torch.mean(torch.log(1 + 1000 * coords_error.pow(2)))

        confs_loss = F.binary_cross_entropy(confs_pred, confs_true_binary)
       
        # Weighted loss
        
        # alpha = 0.05   
        alpha =  1 / 3 

        # alpha = coords_loss.item() / (confs_loss.item() + 1e-8)
        
        alpha_coords_loss = coords_loss
        alpha_confs_loss = alpha * confs_loss
        
        loss = alpha_coords_loss + alpha_confs_loss 
        
        # Calculate Coords metrics

        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=3.0)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2.0)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.0)
        coords_percent_05_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=0.5)

        coords_mae = self.mae(coords_pred, targets)

        # Calculate Confs metrics

        conf_accuracy = self.conf_accuracy(confs_pred_binary, confs_true_binary)

        conf_matrix = self.conf_matrix(confs_pred, confs_true_binary)
        tn, fp, fn, tp = conf_matrix.flatten()
        
        precision_0 = tn / (tn + fn + 1e-8)
        recall_0 = tn / (tn + fp + 1e-8)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + 1e-8)
        
        precision_1 = tp / (tp + fp + 1e-8)
        recall_1 = tp / (tp + fn + 1e-8)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)
        
        # Test
        
        conf_mask = confs_pred > 0.98
        coords_pred_conf = coords_pred[conf_mask]
        targets_conf = targets[conf_mask]

        coords_percent_3_pixel_subset = torch.tensor(0.0, device=coords_pred.device)
        coords_percent_2_pixel_subset = torch.tensor(0.0, device=coords_pred.device)
        coords_percent_15_pixel_subset = torch.tensor(0.0, device=coords_pred.device)
        coords_percent_125_pixel_subset = torch.tensor(0.0, device=coords_pred.device)
        coords_percent_1_pixel_subset = torch.tensor(0.0, device=coords_pred.device)
        coords_percent_05_pixel_subset = torch.tensor(0.0, device=coords_pred.device)

        if coords_pred_conf.numel() > 0:
            coords_percent_3_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=3.0)
            coords_percent_2_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=2.0)
            coords_percent_15_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=1.5)
            coords_percent_125_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=1.25)
            coords_percent_1_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=1.0)
            coords_percent_05_pixel_subset = self._compute_coords_accuracy_percentage(coords_pred_conf, targets_conf, pixels=0.5)

        # Collect

        # clamp for visualization and further usage
        coords_pred = torch.clamp(coords_pred, min=0.0, max=float(config.image.patch_size))

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/coords_loss": coords_loss,
            f"{stage}/conf_loss": confs_loss, 
            f"{stage}/alpha_coords_loss": alpha_coords_loss, 
            f"{stage}/alpha_confs_loss": alpha_confs_loss, 

            f"{stage}/coords_mae": coords_mae,
            f"{stage}/coords_percent_3_pixel": coords_percent_3_pixel,
            f"{stage}/coords_percent_2_pixel": coords_percent_2_pixel,
            f"{stage}/coords_percent_15_pixel": coords_percent_15_pixel,
            f"{stage}/coords_percent_125_pixel": coords_percent_125_pixel,
            f"{stage}/coords_percent_1_pixel": coords_percent_1_pixel,
            f"{stage}/coords_percent_05_pixel": coords_percent_05_pixel,

            f"{stage}/coords_percent_3_pixel_subset": coords_percent_3_pixel_subset,
            f"{stage}/coords_percent_2_pixel_subset": coords_percent_2_pixel_subset,
            f"{stage}/coords_percent_15_pixel_subset": coords_percent_15_pixel_subset,
            f"{stage}/coords_percent_125_pixel_subset": coords_percent_125_pixel_subset,
            f"{stage}/coords_percent_1_pixel_subset": coords_percent_1_pixel_subset,
            f"{stage}/coords_percent_05_pixel_subset": coords_percent_05_pixel_subset,
            
            f"{stage}/conf_accuracy": conf_accuracy,

            f"{stage}/conf_precision_0": precision_0.float(),
            f"{stage}/conf_recall_0": recall_0.float(),
            f"{stage}/conf_f1_0": f1_0.float(),
            f"{stage}/conf_precision_1": precision_1.float(),
            f"{stage}/conf_recall_1": recall_1.float(),
            f"{stage}/conf_f1_1": f1_1.float(),

            f"{stage}/conf_tp": tp.float(),
            f"{stage}/conf_fp": fp.float(),
            f"{stage}/conf_fn": fn.float(),
            f"{stage}/conf_tn": tn.float(),
        }

        return metrics, coords_pred, confs_pred

    def training_step(self, batch, batch_idx):
        metrics, coords_pred, confs_box_pred, confs_point_pred = self._shared_step(batch, stage='train')
        conf_combined = confs_box_pred * confs_point_pred

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_combined, limit_count=limit_count, stage='train')

        loss = metrics['train/loss']

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        metrics, coords_pred, confs_box_pred, confs_point_pred = self._shared_step(batch, stage='val')
        conf_combined = confs_box_pred * confs_point_pred
        
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_combined, limit_count=limit_count, stage='val')
        
        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        metrics, coords_pred, confs_box_pred, confs_point_pred = self._shared_step(batch, stage='test')
        conf_combined = confs_box_pred * confs_point_pred
        
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_combined, limit_count=limit_count, stage='val')
        
        return metrics

    def _log_images(self, batch, target_coords, confidence_pred=None, limit_count=None, stage=None):
        ref_patches, references, tar_patches, targets, estimates, confidences = batch
                
        if limit_count is not None:
            mae_per_patch = torch.abs(target_coords - targets.squeeze(1)).mean(dim=1)
            worst_indices = torch.argsort(mae_per_patch, descending=True)[:limit_count]
            
            ref_patches = ref_patches[worst_indices]
            tar_patches = tar_patches[worst_indices]
            references = references[worst_indices]
            targets = targets[worst_indices]
            estimates = estimates[worst_indices]
            target_coords = target_coords[worst_indices]
            confidence_pred = confidence_pred[worst_indices] if confidence_pred is not None else None
            confidences = confidences[worst_indices]
        
        image_grid = show_batch(
            ref_patches, tar_patches,
            references, 

            patch_level_target_coords=target_coords, patch_level_target_coords_true=targets,

            confidences_true=confidences,
            confidence_pred=confidence_pred,

            estimates=estimates,

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
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # weight_decay=0.1

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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

        progress_bar_callback = TQDMProgressBar(refresh_rate=50, leave=False) 
        summary_callback = ModelSummary(max_depth=1)

        callbacks = [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            summary_callback,
        ]
        
        if len(config.loggers):
            lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor_callback)
        
        return callbacks 

    @staticmethod
    def _normalize_coords(coords):
        """Normalizes coordinates from [0, 81] to [-1, 1] in a numerically stable way."""
        return (coords - (config.image.patch_size - 1) / 2) / ((config.image.patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 81]."""
        return coords * ((config.image.patch_size - 1) / 2) + (config.image.patch_size - 1) / 2

