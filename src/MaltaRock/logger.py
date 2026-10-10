"""Console logging for the command-line interface.

Library modules only use ``logging.getLogger(__name__)``; handlers are
installed by :func:`setup_logging` (called by the CLI) or by the user.
"""

from __future__ import annotations

import logging

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Attach a Rich console handler to the ``MaltaRock`` logger."""
    from rich.logging import RichHandler

    logger = logging.getLogger("MaltaRock")
    if not any(isinstance(handler, RichHandler) for handler in logger.handlers):
        logger.addHandler(RichHandler(rich_tracebacks=True, log_time_format="[%X]"))
    logger.setLevel(level.upper())
    return logger
