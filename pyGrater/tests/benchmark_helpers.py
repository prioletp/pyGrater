"""Small helpers shared by pyGrater benchmark scripts."""
import logging

import os



from pyGrater.config.logging_config import log_info
logger = logging.getLogger(__name__)
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


def rss_mb():
    """Return current process resident memory in MiB, or NaN without psutil."""
    if not HAS_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def divider(title=""):
    """Print a compact benchmark-section divider."""
    width = 70
    if title:
        log_info(logger, f"\n{'─' * 3} {title} {'─' * (width - 5 - len(title))}")
    else:
        log_info(logger, "─" * width)
