import logging
from typing import Optional


def setup_logger(name: Optional[str] = None):
    logger = logging.getLogger(name or __name__)
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.level = logging.INFO
    return logger
