import logging
from threading import Lock
from typing import Optional



class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[32m",
        "INFO": "\033[37m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"


_logger_lock = Lock()
_logger: logging.Logger = None


def _init_logger(level="INFO") -> logging.Logger:
    global _logger
    with _logger_lock:
        if _logger is None:
            logger = logging.getLogger("myapp")

   
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = ColorFormatter(
                    "%(asctime)s | %(levelname)-6s | %(message)s", "%Y-%m-%d %H:%M:%S"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.propagate = False
            # -------------------------------------------------

            logger.setLevel(level)
            logging.getLogger().setLevel(logging.WARNING)
            _logger = logger
        else:
            _logger.setLevel(level)
    return _logger


logger = _init_logger("INFO")

class BaseConfig:
    MAX_WORKERS: int = 6 

    def __init__(self, log_level: str = "INFO", max_workers: Optional[int] = None):
        if max_workers is not None:
            BaseConfig.MAX_WORKERS = max_workers
        logger.setLevel(log_level)
        self.logger = logger 
