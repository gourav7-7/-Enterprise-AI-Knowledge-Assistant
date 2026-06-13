from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name:str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)

    logger.propagate = False
    return logger