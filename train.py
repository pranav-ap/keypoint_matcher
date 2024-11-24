import lightning.pytorch as pl
import torch

from config import config
from src import Light, MatchesDataModule
from utils import logger, make_clear_directory, MyLogger

torch.set_float32_matmul_precision('medium')


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    loggers = MyLogger.get_loggers()
    neptune_logger, tensorboard_logger = loggers
    # tensorboard_logger = MyLogger.get_loggers()

    # light = Light(
    #     neptune_logger=neptune_logger,
    #     tensorboard_logger=tensorboard_logger
    # )

    checkpoint_path = './output/checkpoints/best_checkpoint.ckpt'
    light = Light.load_from_checkpoint(
        checkpoint_path,
        neptune_logger=neptune_logger,
        tensorboard_logger=tensorboard_logger,
    )

    dm = MatchesDataModule()

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
    )

    trainer.fit(light, datamodule=dm)

    if trainer.checkpoint_callback.best_model_path:
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
