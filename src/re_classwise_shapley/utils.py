import os
from contextlib import contextmanager

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
