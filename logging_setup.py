# utils/logging_setup.py
import logging
import config

def setup_logging():
    """Configures global logging based on settings in config.py."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )