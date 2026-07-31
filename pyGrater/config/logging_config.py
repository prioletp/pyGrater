"""Logging helpers for pyGrater.

pyGrater uses normal Python logging.  Importing the package configures a
console INFO handler for the top-level ``pyGrater`` logger if the application
has not already configured logging.  File logging is opt-in.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """Small colored formatter for interactive console output."""

    cyan = "\x1b[96;20m"
    green = "\x1b[92;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[91;20m"
    bold_red = "\x1b[91;1m"
    reset = "\x1b[0m"
    message_format = "%(colored_levelname)s - %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: cyan,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        colored_levelname = f"{color}{record.levelname}{self.reset}"
        if getattr(record, "pygrater_banner", False):
            width = getattr(record, "pygrater_banner_width", 60)
            line = "=" * width
            return f"{line}\n{colored_levelname} - {record.getMessage()}\n{line}"
        record.colored_levelname = colored_levelname
        formatter = logging.Formatter(self.message_format)
        return formatter.format(record)


def configure_logging(
        level=logging.INFO,
        log_to_file=False,
        log_dir=None,
        logger_name="pyGrater"):
    """Configure and return the top-level pyGrater logger.

    This function never redirects ``sys.stdout``.  It only attaches logging
    handlers to the requested logger when no handler already exists.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, "_pygrater_console", False)
               for handler in logger.handlers):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler._pygrater_console = True
        console_handler.setLevel(level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

    if log_to_file:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pyGrater_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s "
            "(%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file)

    return logger


def setup_logger(name="pyGrater", level=logging.INFO, log_to_file=False,
                 log_dir=None):
    """Backward-compatible alias for configuring a pyGrater logger."""
    return configure_logging(
        level=level,
        log_to_file=log_to_file,
        log_dir=log_dir,
        logger_name=name,
    )


def log_info(logger, *values, sep=" ", end="\n", file=None, flush=False):
    """Log a former ``print`` message at INFO level.

    ``file`` and ``flush`` are accepted for compatibility with old print calls;
    messages still go through the provided logger.
    """
    del end, file, flush
    logger.info(sep.join(str(value) for value in values))


def log_banner(logger, title, width=60):
    """Log a visual banner with the INFO label only on the title line."""
    logger.info(
        title,
        extra={"pygrater_banner": True, "pygrater_banner_width": width})


if __name__ == "__main__":
    logger = configure_logging(level=logging.DEBUG, log_to_file=False)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
