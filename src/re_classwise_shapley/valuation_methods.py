import logging
import math as m
from typing import Literal

from pydvl.parallel.config import ParallelConfig
from pydvl.utils import Utility
from pydvl.value import (
    ClasswiseScorer,
    MaxChecks,
    MinUpdates,
    RelativeTruncation,
    ShapleyMode,
    ValuationResult,
    compute_classwise_shapley_values,
    compute_least_core_values,
    compute_loo,
    compute_shapley_values,
)
from pydvl.value.semivalues import SemiValueMode, compute_semivalues

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import n_threaded

logger = setup_logger(__name__)


def compute_values(
    utility: Utility,
    valuation_method: Literal[
        "random",
        "loo",
        "classwise_shapley",
        "beta_shapley",
        "banzhaf_shapley",
        "tmc_shapley",
        "owen_sampling_shapley",
        "least_core",
    ],
    seed: int = None,
    **kwargs,
) -> ValuationResult:
    """
    Computes the valuation values for a given valuation method. The valuation method is
    specified by the `valuation_method` argument. The `kwargs` are passed to the
    valuation method.

    TODO Remove this method by integrating function calls in to the params.yaml file.

    Args:
        utility: Utility object to compute the valuation values for.
        valuation_method: A method to compute the valuation values.
        seed: Either a seed or a seed sequence to use for the random number generator.
        **kwargs:

    Returns:
        The valuation values for the given valuation method.
    """
    match valuation_method:
        case "random":
            return ValuationResult.from_random(size=len(utility.data), seed=seed)

    n_jobs = kwargs["n_jobs"]
    parallel_config = ParallelConfig(
        backend=kwargs["backend"],
        n_cpus_local=n_jobs,
        logging_level=logging.INFO,
    )
    progress = kwargs.get("progress", False)
    with n_threaded():
        match valuation_method:
            case "loo":
                return compute_loo(utility, n_jobs=n_jobs, progress=progress)
            case "classwise_shapley":
                n_updates = int(kwargs.get("n_updates"))
                utility.scorer = ClasswiseScorer("accuracy", default=0.0)
                return compute_classwise_shapley_values(
                    utility,
                    done=MinUpdates(n_updates=n_updates),
                    truncation=RelativeTruncation(utility, rtol=float(kwargs["rtol"])),
                    normalize_values=kwargs["normalize_values"],
                    done_sample_complements=MaxChecks(
                        kwargs["n_resample_complement_sets"]
                    ),
                    use_default_scorer_value=kwargs.get(
                        "use_default_scorer_value", True
                    ),
                    min_elements_per_label=kwargs.get("min_elements_per_label", 1),
                    n_jobs=n_jobs,
                    config=parallel_config,
                    progress=progress,
                    seed=seed,
                )
            case "beta_shapley":
                n_updates = int(kwargs.get("n_updates"))
                return compute_semivalues(
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
            case "banzhaf_shapley":
                n_updates = int(kwargs.get("n_updates"))
                return compute_semivalues(
                    u=utility,
                    mode=SemiValueMode.Banzhaf,
                    done=MinUpdates(n_updates=n_updates),
                    batch_size=len(utility.data),
                    n_jobs=n_jobs,
                    config=parallel_config,
                    progress=progress,
                    seed=seed,
                )

            case "tmc_shapley":
                n_updates = int(kwargs.get("n_updates"))
                return compute_shapley_values(
                    utility,
                    mode=ShapleyMode.PermutationMontecarlo,
                    truncation=RelativeTruncation(utility, rtol=float(kwargs["rtol"])),
                    done=MinUpdates(n_updates=n_updates),
                    n_jobs=n_jobs,
                    config=parallel_config,
                    progress=progress,
                    seed=seed,
                )

            case "owen_sampling_shapley":
                n_updates = int(kwargs.get("n_updates"))
                n_updates = int(m.ceil(n_updates / n_jobs))
                return compute_shapley_values(
                    utility,
                    mode=ShapleyMode.Owen,
                    n_jobs=n_jobs,
                    progress=progress,
                    seed=seed,
                    n_samples=n_updates,
                    max_q=kwargs.get("max_q"),
                )
            case "least_core":
                n_updates = int(kwargs.get("n_updates"))
                return compute_least_core_values(
                    utility, n_iterations=n_updates, n_jobs=n_jobs, progress=progress
                )

            case _:
                raise NotImplementedError(
                    f"The method {valuation_method} is not registered within."
                )
