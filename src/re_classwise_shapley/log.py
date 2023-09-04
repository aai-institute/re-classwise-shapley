import logging
from typing import Optional


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with a uniform configuration.
    :param name: Optional name of the logger. If no name is given, the name of the
        current module is used.
    :return: A logger with the given name and hard coded configuration.
    """
    logger = logging.getLogger(name or __name__)
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.level = logging.DEBUG
    return logger
