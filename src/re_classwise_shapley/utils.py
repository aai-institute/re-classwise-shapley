import os
from contextlib import contextmanager
from typing import Dict

import yaml
from pydvl.utils import Seed, ensure_seed_sequence

from re_classwise_shapley.log import setup_logger

logger = setup_logger()

__all__ = ["pipeline_seed", "n_threaded"]


def pipeline_seed(initial_seed: Seed, pipeline_step: int) -> int:
    """
    Get the seed for a specific pipeline step. The seed is generated from the initial
    seed and the pipeline step.

    Args:
        initial_seed: Initial seed.
        pipeline_step: Pipeline step.

    Returns:
        The seed for the given pipeline step.
    """
    return int(ensure_seed_sequence(initial_seed).generate_state(pipeline_step)[-1])


@contextmanager
def n_threaded(n_threads: int = 1) -> None:
    """
    Context manager to temporarily set the number of threads for numpy, scipy and
    pytorch. This is necessary to avoid over-subscription of threads.

    Args:
        n_threads: Number of threads to use.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    yield None
    del os.environ["OMP_NUM_THREADS"]
    del os.environ["OPENBLAS_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]
    del os.environ["VECLIB_MAXIMUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]


def flatten_dict(d: Dict, parent_key: str = "", separator: str = ".") -> Dict:
    """
    Flatten a nested dictionary. Recursively add the values under a new key that is
    constructed by concatenating the parent key and the current key with the separator.

    Args:
        d: Dictionary to flatten.
        parent_key: Parent key to use for the new key.
        separator: Separator to use for the new key.

    Returns:
        Flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, separator=separator))
        else:
            items[new_key] = v
    return items


def load_params_fast() -> Dict:
    """
    Load the parameters from the `params.yaml` file without verification.
    """
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)
