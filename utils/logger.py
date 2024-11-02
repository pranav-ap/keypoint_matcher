from config import config
import sys
from loguru import logger
import warnings


def setup_logging():
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.remove()  # Remove the default handler
    logger.add(
        sys.stdout,
        format="<level>{level: <8}</level> | "
               "<cyan>{function}</cyan> | "
               "<level>{message}</level>",
        level=config.log_level,
    )


# Ensure the logger is set up when this module is imported
setup_logging()
