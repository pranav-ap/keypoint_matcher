import lightning.pytorch as pl
import torch

from config import config
from src import Light, MatchesDataModule
from utils import logger

torch.set_float32_matmul_precision('medium')


def main():
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    dm = MatchesDataModule()

    checkpoint_path = ''
    light = Light.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        devices='auto',
        accelerator="auto",
        enable_model_summary=False,
    )

    trainer.predict(light, datamodule=dm)


if __name__ == '__main__':
    main()
