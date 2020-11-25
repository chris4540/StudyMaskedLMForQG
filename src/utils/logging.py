"""
This module should be import at least once

Example
-------
from utils.logging import logging
logger = logging.getLogger(__name__)
...
"""
import logging
import logging.config
from logging import Logger

# -------------------------------------------------------
#            Default Logging config
# -------------------------------------------------------
DEFAULT_CFG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s %(levelname)s: %(message)s",
            'datefmt': '%Y-%m-%d %H:%M'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    }
}


def _reset_logging_config():
    logging.config.dictConfig(DEFAULT_CFG)


_reset_logging_config()

# -------------------------------------------------------
#            Default Logging config
# -------------------------------------------------------


def set_logfile_output(logfile, loglevel=logging.INFO):
    """
    Set the logfile output locaiton
    Args:
        logfile (str): the output logfile path
    """

    # setup format
    fmt = DEFAULT_CFG["formatters"]["standard"]
    formatter = logging.Formatter(
        fmt=fmt["format"], datefmt=fmt["datefmt"])

    # setup file handler
    handler = logging.FileHandler(logfile, encoding="utf-8")
    handler.setFormatter(formatter)

    # Add the prepared handler to root
    root = logging.getLogger()
    root.addHandler(handler)


class DummyLogger(Logger):
    """
    python 3.6 has the problem using mp and logging

    This logger is a dummy logger
    """

    def __init__(self):
        pass

    def __getattr__(self, name):
        return lambda *x: None
