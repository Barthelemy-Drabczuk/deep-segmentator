"""Logging configuration using loguru."""
import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("sulcal_seg")  # type: ignore[assignment]
    HAS_LOGURU = False


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure application-wide logging."""
    if HAS_LOGURU:
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        )
        if log_file is not None:
            logger.add(log_file, rotation="100 MB", level=log_level, retention="7 days")
    else:
        _logging.basicConfig(
            level=getattr(_logging, log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        )


def get_logger(name: str):
    """Get a named logger instance."""
    if HAS_LOGURU:
        return logger.bind(name=name)
    return _logging.getLogger(name)  # type: ignore[return-value]
