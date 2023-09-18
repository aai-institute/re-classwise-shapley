import os
import shutil
from pathlib import Path

from pydvl.utils import Seed, ensure_seed_sequence

from re_classwise_shapley.log import setup_logger

logger = setup_logger()

__all__ = [
    "get_pipeline_seed",
    "clear_folder",
]


def get_pipeline_seed(initial_seed: Seed, pipeline_step: int) -> int:
    return int(ensure_seed_sequence(initial_seed).generate_state(pipeline_step)[-1])


def clear_folder(path: Path):
    """
    Clear the folder at the given path.
    :param path: Path to the folder to clear.
    """
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
