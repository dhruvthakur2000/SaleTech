
import sys
import os
from loguru import logger
from config.settings import AppSettings


def setup_logger():
    settings = AppSettings()
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Console output
    logger.add(
        sys.stdout,
        level="DEBUG" if settings.debug else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    
    # File output
    logger.add(
        "logs/app_{time:YYYY-MM-DD___HH-mm-ss}.log",
        level="DEBUG" if settings.debug else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        retention="7 days",  # Keep logs for 7 days
    )

    return logger
           