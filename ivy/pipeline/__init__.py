import logging
from pathlib import Path

# Exported application logger, safe to import as: from pipeline import logger
logger = logging.getLogger("logger-ivy-agent")

if not logger.handlers:
    try:
        log_file_path = Path(__file__).with_name("logger.log")
        handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            "%(levelname)s:%(filename)s:%(funcName)s:%(asctime)s - %(lineno)i:%(message)s"
        )
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    except Exception as e:  # Fallback: still expose a usable logger
        fallback_handler = logging.NullHandler()
        logger.addHandler(fallback_handler)
        logger.error("Failed to configure file logger: %s", e)