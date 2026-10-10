# type: ignore

import logging
from threading import Lock
from rich.console import Console
from rich.logging import RichHandler


class Logger:
    _instance = None
    _lock = Lock()  # Thread-safe initialization

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # Prevent re-initialization
        self._initialized = True
        self.console = Console()

        # Configure Rich logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
            encoding="utf-8",
        )
        self.logger = logging.getLogger("rich_logger")
        self.logger.setLevel(logging.INFO)  # Default log level

    def get_logger(self) -> logging.Logger:
        return self.logger

    def set_level(self, level: str, is_verbose: bool = False) -> None:
        """Set the logging level dynamically.

        Args:
            level (str): The desired logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """
        self.is_verbose = is_verbose
        level = level.upper()
        if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            self.logger.setLevel(getattr(logging, level))
            self.logger.info(f"Log level set to {level}")
        else:
            self.logger.error(
                f"Invalid log level: {level}. Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
            )
