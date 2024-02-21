import logging
import math as m
import os
import shutil

from pydvl.utils import ClasswiseScorer, ParallelConfig, Utility
from pydvl.value import (
    MaxUpdates,
    RelativeTruncation,
    ShapleyMode,
    ValuationResult,
    compute_classwise_shapley_values,
    compute_shapley_values,
    naive_loo,
)
from pydvl.value.semivalues import SemiValueMode, compute_semivalues

from csshapley22.log import setup_logger

logger = setup_logger(__name__)


def compute_values(
    utility: Utility, valuation_method: str, **kwargs
) -> ValuationResult:
    progress = kwargs.get("progress", False)
    tmp_dir = kwargs["temp_dir"]
    n_jobs = kwargs["n_jobs"]
    parallel_config = ParallelConfig(
        backend=kwargs["backend"],
        n_cpus_local=n_jobs,
        logging_level=logging.INFO,
        _temp_dir=tmp_dir,
    )
    if valuation_method == "random":
        values = ValuationResult.from_random(size=len(utility.data))

    elif valuation_method == "loo":
        values = naive_loo(utility, progress=progress)

    elif valuation_method == "classwise_shapley":
        utility.scorer = ClasswiseScorer("accuracy", default=0.0)
        values = compute_classwise_shapley_values(
            utility,
            done=MaxUpdates(n_updates=int(kwargs["n_updates"])),
            truncation=RelativeTruncation(utility, rtol=kwargs["rtol"]),
            normalize_values=kwargs["normalize_values"],
            n_resample_complement_sets=kwargs["n_resample_complement_sets"],
            use_default_scorer_value=kwargs.get("use_default_scorer_value", True),
            min_elements_per_label=kwargs.get("min_elements_per_label", 1),
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
        )

    elif valuation_method == "beta_shapley":
        values = compute_semivalues(
            u=utility,
            mode=SemiValueMode.BetaShapley,
            done=MaxUpdates(n_updates=int(kwargs["n_updates"])),
            alpha=kwargs["alpha"],
            beta=kwargs["beta"],
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
        )

    elif valuation_method == "tmc_shapley":
        values = compute_shapley_values(
            utility,
            mode=ShapleyMode.PermutationMontecarlo,
            truncation=RelativeTruncation(utility, rtol=kwargs["rtol"]),
            done=MaxUpdates(n_updates=int(kwargs["n_updates"])),
            n_jobs=n_jobs,
            config=parallel_config,
            progress=progress,
        )

    else:
        raise NotImplementedError(
            f"The method {valuation_method} is not registered within."
        )

    logger.info(f"Values: {values.values}")
    return values


def clear_folder(path: str):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
