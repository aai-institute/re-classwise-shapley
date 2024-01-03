import logging
import math as m
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from pydvl.parallel.config import ParallelConfig
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility
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
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.types import Seed
from re_classwise_shapley.utils import load_params_fast, n_threaded

__all__ = ["compute_values", "calculate_subset_score"]

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


class SubsetScorer(Scorer):
    """
    A scorer which operates on a subset and additionally normalizes the output score.

    Args:
        subset: An array of indices mapping to the subset of training indices to include in the score calculation.
        normalize: True, iff the score shall be multiplied by `len(subset) / len(train_indices)`.
    """

    def __init__(
        self, *args, subset: NDArray[np.int_], normalize: bool = True, **kwargs
    ):
        Scorer.__init__(self, *args, **kwargs)
        self._idx = subset
        self._normalize = normalize

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        n = len(y)
        idx = self._idx
        score = Scorer.__call__(self, model=model, X=X[idx], y=y[idx])
        return score * len(idx) / n


def calculate_subset_score(
    data_set: Dataset,
    subset_idx_fn: Callable[[int], NDArray[np.int_]],
    model_name: str,
    model_seed: Seed,
    sampler_seed: Seed,
    valuation_method_name: str,
    n_jobs: int,
    backend,
):
    """Calculates the subset score for a given dataset and model.

    This function evaluates the performance of a specified machine learning model on subsets of a given dataset. It uses
    a unique subset indexing function to select different subsets and computes their scores using a specified valuation
    method.

    Args:
        data_set: The dataset on which the model is evaluated.
        subset_idx_fn: A function that takes an integer and returns a subset of indices from the dataset.
        model_name: The name of the machine learning model to be used.
        model_seed: A seed for initializing the model to ensure reproducibility.
        sampler_seed: A seed for the sampling process.
        valuation_method_name: The name of the valuation method to be used for scoring.
        n_jobs: The number of jobs to run in parallel during computation.
        backend: The backend to be used for computation.

    Returns:
        A tuple containing the marginal accuracy scores for each subset (`subset_mar_acc`) and a dictionary with
            statistical data (`subset_stats`) like mean and standard deviation of these scores.
    """
    params = load_params_fast()
    valuation_method_config = params["valuation_methods"][valuation_method_name]
    all_classes = np.unique(data_set.y_train)
    subset_mar_acc = []
    for c in all_classes:
        params = load_params_fast()
        model_kwargs = params["models"][model_name]
        u = Utility(
            data=data_set,
            model=instantiate_model(model_name, model_kwargs, seed=int(model_seed)),
            scorer=SubsetScorer("accuracy", subset=subset_idx_fn(c), default=np.nan),
            catch_errors=False,
        )
        values = compute_values(
            u,
            valuation_method_name,
            **valuation_method_config,
            backend=backend,
            n_jobs=n_jobs,
            seed=sampler_seed,
        )
        subset_mar_acc.append(values.values)

    subset_mar_acc = np.stack(subset_mar_acc, axis=1)
    subset_mar_acc = np.take_along_axis(
        subset_mar_acc, data_set.y_train.reshape([-1, 1]), axis=1
    ).reshape(-1)
    subset_stats = {
        "mean": np.mean(subset_mar_acc),
        "std": np.std(subset_mar_acc),
    }
    return subset_mar_acc, subset_stats
