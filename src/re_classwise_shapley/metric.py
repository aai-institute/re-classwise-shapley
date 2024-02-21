from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from dvc.api import params_show
from numpy.typing import NDArray
from pydvl.parallel import (
    ParallelConfig,
    effective_n_jobs,
    init_executor,
    init_parallel_backend,
)
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility, maybe_progress
from pydvl.value.result import ValuationResult
from sklearn.metrics import auc
from tqdm import tqdm

from re_classwise_shapley.log import setup_logger

__all__ = [
    "metric_weighted_metric_drop",
    "metric_roc_auc",
]

from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger(__name__)


def metric_roc_auc(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    flipped_labels: str,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> Tuple[float, pd.Series]:
    """
    Computes the area under the ROC curve for a given valuation result on a dataset.

    Args:
        data: Dataset to compute the area under the ROC curve on.
        values: Valuation result to compute the area under the ROC curve for.
        info: Dictionary containing information about the dataset.
        flipped_labels: Name of the key in the info dictionary containing the indices of
            the flipped labels.
        n_jobs: Number of parallel jobs to run.
        config: Parallel configuration.
        progress: Whether to display a progress bar.

    Returns:
        The area under the ROC curve.
    """
    ranked_list = list(np.argsort(values.values))
    ranked_list = values.indices[ranked_list]
    precision_recall = _curve_precision_recall_ranking(
        info[flipped_labels], ranked_list
    )
    return auc(precision_recall.index, precision_recall.values), precision_recall


def metric_weighted_metric_drop(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    eval_model: str,
    metric: str = "accuracy",
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> Tuple[float, pd.Series]:
    r"""
    Calculates the weighted reciprocal difference average of a valuation function. The
    formula

    $$
    \frac{1}{n} \sum_{i=1}^{n} \frac{1}{i} \sum_{j=1}^{i} \Delta_j
    $$

    averages the difference between the successive metric scores of the valuation.

    Args:
        data: Dataset to compute the weighted reciprocal difference average on.
        values: Valuation result to compute the weighted reciprocal difference average
        info: Additional information about the dataset.
        eval_model: Evaluation model to use for evaluation.
        metric: Scorer metric to use for evaluation.
        n_jobs: Number of parallel jobs to run.
        config: Parallel configuration.
        progress: Whether to display a progress bar.

    Returns:
        The weighted reciprocal difference average in the given metric.
    """
    model_kwargs = load_params_fast()["models"][eval_model]
    model = instantiate_model(eval_model, model_kwargs, seed=0)
    u_eval = Utility(
        data=data,
        model=model,
        scorer=Scorer(metric, default=np.nan),
        catch_errors=True,
    )
    curve = _curve_score_over_point_removal_or_addition(
        u_eval,
        values,
        highest_point_removal=True,
        progress=progress,
        n_jobs=n_jobs,
        config=config,
    )
    curve.name = metric
    diff_curve = curve.diff(-1).values[:-1]
    diff_curve = np.nancumsum(diff_curve)
    weighted_diff = diff_curve / (np.arange(1, len(diff_curve) + 1))
    return float(np.sum(weighted_diff)), curve


def _curve_precision_recall_ranking(
    target_list: NDArray[np.int_], ranked_list: NDArray[np.int_]
) -> pd.Series:
    """
    Computes the precision-recall curve for a given ranking. The ranking is given as a
    list of indices. The target list is a list of indices of the target class. For each
    prefix of the ranking, the precision and recall for the ground truth are computed.

    Args:
        target_list: List of indices of the ground truth.
        ranked_list: List of indices of the ranking in descending order.

    Returns:
        A pd.Series object containing the precision and recall values for each prefix of
        the ranking. The index of the series is the recall and the values are the
        corresponding precision values.
    """
    precision, recall = np.zeros(len(ranked_list)), np.zeros(len(ranked_list))
    for idx in range(len(ranked_list)):
        partial_list = ranked_list[: idx + 1]
        intersection = list(set(target_list) & set(partial_list))
        precision[idx] = float(len(intersection) / len(partial_list))
        recall[idx] = float(len(intersection) / len(target_list))

    graph = pd.Series(precision, index=recall)
    graph = graph[~graph.index.duplicated(keep="first")]
    graph = graph.sort_index(ascending=True)
    graph.index.name = "recall"
    graph.name = "precision"
    return graph


def _curve_score_over_point_removal_or_addition(
    u: Utility,
    values: ValuationResult,
    *,
    highest_point_removal: bool = False,
    min_points: int = 6,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> pd.Series:
    """
    Computes the utility score over the removal or addition of points. The utility score
    is computed for each prefix of the ranking. The prefix is either the highest or the
    lowest points of the ranking with respect to the valuation result.

    Args:
        u: Utility to compute the utility score with.
        values: Valuation result to rank the data points.
        highest_point_removal: True, if the highest points should be removed. False, if
            the lowest points should be added.
        min_points: Minimum number of points to keep in the set.
        n_jobs: Number of parallel jobs to run.
        config: Parallel configuration.
        progress: Whether to display a progress bar.

    Returns:
        A pd.Series object containing the utility scores for each prefix of the ranking.
        The index of the series is the number of points removed or added and the values
        are the corresponding utility scores.
    """
    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values)}, should be equal to the number of"
            f" data indices, {len(u.data.indices)}"
        )

    parallel_backend = init_parallel_backend(config)
    u_ref = parallel_backend.put(u)
    values_ref = parallel_backend.put(values)
    n_evals = len(u.data) - min_points + 1
    n_jobs = effective_n_jobs(n_jobs, config)
    max_jobs = 2 * n_jobs
    n_submitted_jobs = 0
    n_completed = 0

    pbar = tqdm(disable=not progress, position=0, total=n_evals, unit="%")
    values.sort(reverse=highest_point_removal)
    scores = pd.Series(index=np.arange(n_evals), dtype=np.float64)

    def _evaluate_at_point(
        u: Utility,
        values: ValuationResult,
        j: int,
        highest_point_removal: bool,
        min_points: int,
    ) -> Tuple[int, float]:
        val = (
            u(values.indices[j:])
            if highest_point_removal
            else u(values.indices[: min_points + j])
        )
        return j, val

    with init_executor(max_workers=n_jobs, config=config) as executor:
        pending: Set[Future] = set()
        while True:
            completed_futures, pending = wait(
                pending, timeout=60, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                idx, score = future.result()
                n_completed += 1
                pbar.n = n_completed
                pbar.refresh()
                scores.iloc[idx] = score

            if n_completed == n_evals:
                break
            elif n_submitted_jobs < n_evals:
                rem_jobs = n_evals - n_submitted_jobs
                n_remaining_slots = min(rem_jobs, max_jobs) - len(pending)
                for i in range(n_remaining_slots):
                    future = executor.submit(
                        _evaluate_at_point,
                        u_ref,
                        values_ref,
                        j=n_submitted_jobs,
                        min_points=min_points,
                        highest_point_removal=highest_point_removal,
                    )
                    n_submitted_jobs += 1
                    pending.add(future)

    if not highest_point_removal:
        scores.index = scores.index.values + min_points

    scores.index.name = "n_points_" + ("removed" if highest_point_removal else "added")
    return scores
