from config import config

import lightning.pytorch as pl
import torch

from src import Light, MatchesDataModule
from utils import logger, make_clear_directory, MyLogger

torch.set_float32_matmul_precision('medium')


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    neptune_logger, tensorboard_logger = MyLogger.neptune_logger, MyLogger.tensorboard_logger

    loggers = []
    if neptune_logger is not None:
        loggers.append(neptune_logger)
    if tensorboard_logger is not None:
        loggers.append(tensorboard_logger)

    # checkpoint_path = '/home/stud/ath/ath_ws/keypoint_matcher/output/smallnet_depthwise/checkpoints/last.ckpt'
    # light = Light.load_from_checkpoint(
    #     checkpoint_path,
    #     neptune_logger=neptune_logger,
    #     tensorboard_logger=tensorboard_logger,
    # )

    light = Light(
        neptune_logger=neptune_logger,
        tensorboard_logger=tensorboard_logger
    )

    dm = MatchesDataModule()

    if neptune_logger is not None:
        neptune_logger.log_model_summary(model=light, max_depth=-1)

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=loggers,
        devices='auto',
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        log_every_n_steps=config.train.log_every_n_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        fast_dev_run=config.train.fast_dev_run,
        overfit_batches=config.train.overfit_batches,
        enable_model_summary=False,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        # precision="16-mixed",
    )

    trainer.fit(light, datamodule=dm)

    # noinspection PyUnresolvedReferences
    if trainer.checkpoint_callback.best_model_path:
        # noinspection PyUnresolvedReferences
        logger.info(f"Best model path : {trainer.checkpoint_callback.best_model_path}")


def prep_directories():
    logger.info("Clearing Directories")
    make_clear_directory(config.paths.output.train_images)
    make_clear_directory(config.paths.output.val_images)
    make_clear_directory(config.paths.output.checkpoints)


def main():
    torch.cuda.empty_cache()
    MyLogger.init_loggers()
    prep_directories()

    train()


if __name__ == '__main__':
    main()
