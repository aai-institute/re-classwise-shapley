import logging
import math as m
import os

from pydvl.utils import ParallelConfig, Utility
from pydvl.value import (
    ClasswiseScorer,
    MaxChecks,
    MaxUpdates,
    MinUpdates,
    RelativeTruncation,
    ShapleyMode,
    ValuationResult,
    compute_classwise_shapley_values,
    compute_least_core_values,
    compute_loo,
    compute_shapley_values,
    owen_sampling_shapley,
)
from pydvl.value.semivalues import SemiValueMode, compute_semivalues

from re_classwise_shapley.log import setup_logger

logger = setup_logger(__name__)


def set_num_threads(n_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def unset_threads():
    del os.environ["OMP_NUM_THREADS"]
    del os.environ["OPENBLAS_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]
    del os.environ["VECLIB_MAXIMUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]


def compute_values(
    utility: Utility,
    valuation_method: str,
    *,
    seed: int = None,
    **kwargs,
) -> ValuationResult:
    if valuation_method == "random":
        return ValuationResult.from_random(size=len(utility.data), seed=seed)

    n_jobs = kwargs["n_jobs"]
    parallel_config = ParallelConfig(
        backend=kwargs["backend"],
        n_cpus_local=n_jobs,
        logging_level=logging.INFO,
    )
    progress = kwargs.get("progress", False)
    set_num_threads(1)

    if valuation_method == "loo":
        values = compute_loo(utility, n_jobs=n_jobs, progress=progress)

    elif valuation_method == "classwise_shapley":
        n_updates = int(kwargs.get("n_updates"))
        utility.scorer = ClasswiseScorer("accuracy", default=0.0)
        values = compute_classwise_shapley_values(
            utility,
            done=MinUpdates(n_updates=n_updates),
            truncation=RelativeTruncation(utility, rtol=kwargs["rtol"]),
            normalize_values=kwargs["normalize_values"],
            done_sample_complements=MaxChecks(kwargs["n_resample_complement_sets"]),
            use_default_scorer_value=kwargs.get("use_default_scorer_value", True),
            min_elements_per_label=kwargs.get("min_elements_per_label", 1),
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
            seed=seed,
        )

    elif valuation_method == "beta_shapley":
        n_updates = int(kwargs.get("n_updates"))
        values = compute_semivalues(
            u=utility,
            mode=SemiValueMode.BetaShapley,
            done=MinUpdates(n_updates=n_updates),
            batch_size=len(utility.data),
            alpha=kwargs["alpha"],
            beta=kwargs["beta"],
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
            seed=seed,
        )

    elif valuation_method == "banzhaf_shapley":
        n_updates = int(kwargs.get("n_updates"))
        values = compute_semivalues(
            u=utility,
            mode=SemiValueMode.Banzhaf,
            done=MinUpdates(n_updates=n_updates),
            batch_size=len(utility.data),
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
            seed=seed,
        )

    elif valuation_method == "tmc_shapley":
        n_updates = int(kwargs.get("n_updates"))
        values = compute_shapley_values(
            utility,
            mode=ShapleyMode.PermutationMontecarlo,
            truncation=RelativeTruncation(utility, rtol=kwargs["rtol"]),
            done=MinUpdates(n_updates=n_updates),
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
            seed=seed,
        )

    elif valuation_method == "owen_sampling_shapley":
        n_updates = int(kwargs.get("n_updates"))
        n_updates = int(m.ceil(n_updates / n_jobs))
        values = compute_shapley_values(
            utility,
            mode=ShapleyMode.Owen,
            n_jobs=n_jobs,
            progress=progress,
            seed=seed,
            n_samples=n_updates,
            max_q=kwargs.get("max_q"),
        )
    elif valuation_method == "least_core":
        n_updates = int(kwargs.get("n_updates"))
        values = compute_least_core_values(
            utility, n_iterations=n_updates, n_jobs=n_jobs, progress=progress
        )

    else:
        raise NotImplementedError(
            f"The method {valuation_method} is not registered within."
        )

    unset_threads()
    logger.info(f"Values: {values.values}")
    return values
