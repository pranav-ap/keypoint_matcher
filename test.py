import lightning.pytorch as pl
import torch

from config import config
from src import Light, MatchesDataModule
from utils import logger, make_clear_directory, MyLogger

torch.set_float32_matmul_precision('medium')


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    neptune_logger, tensorboard_logger = MyLogger.neptune_logger, MyLogger.tensorboard_logger

    loggers = []
    if neptune_logger is not None:
        loggers.append(neptune_logger)
    if tensorboard_logger is not None:
        loggers.append(tensorboard_logger)

    dm = MatchesDataModule()

    checkpoint_path = '/home/stud/ath/ath_ws/keypoint_matcher/output/finalnet_resnet_52_linux_20mil_2_heads/checkpoints/best_checkpoint copy.ckpt'
    light = Light.load_from_checkpoint(
        checkpoint_path,
        neptune_logger=neptune_logger,
        tensorboard_logger=tensorboard_logger,
    )

    if neptune_logger is not None:
        neptune_logger.log_model_summary(model=light, max_depth=-1)

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=loggers,
        devices='auto',
        accelerator="auto",
        log_every_n_steps=config.train.log_every_n_steps,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    trainer.test(light, datamodule=dm)


def main():
    torch.cuda.empty_cache()
    MyLogger.init_loggers()

    test()


if __name__ == '__main__':
    main()
