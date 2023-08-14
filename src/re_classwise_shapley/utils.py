import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from numpy.random import SeedSequence

from re_classwise_shapley.log import setup_logger

logger = setup_logger()

__all__ = [
    "init_random_seed",
    "clear_folder",
]


def init_random_seed(seed: int) -> SeedSequence:
    """Taken verbatim from:
    https://koustuvsinha.com//practices_for_reproducibility/
    """
    seed_sequence = SeedSequence(seed)
    rng = np.random.default_rng(seed_sequence)
    seed = rng.integers(2**31 - 1, size=1)[0]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed_sequence.spawn(1)[0]


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
