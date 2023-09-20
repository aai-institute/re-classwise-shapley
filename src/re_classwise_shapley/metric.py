from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility, maybe_progress
from pydvl.value.result import ValuationResult
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsClassifier

from re_classwise_shapley.log import setup_logger

__all__ = [
    "metric_weighted_metric_drop",
    "curve_score_over_point_removal_or_addition",
    "curve_precision_recall_valuation_result",
    "metric_roc_auc",
]


logger = setup_logger(__name__)


def metric_roc_auc(
    data: Dataset, values: ValuationResult, info: Dict, flipped_labels: str
) -> float:
    """
    Computes the area under the ROC curve for a given valuation result on a dataset.

    Args:
        data: Dataset to compute the area under the ROC curve on.
        values: Valuation result to compute the area under the ROC curve for.
        info: Dictionary containing information about the dataset.
        flipped_labels: Name of the key in the info dictionary containing the indices of
            the flipped labels.

    Returns:
        The area under the ROC curve.
    """
    ranked_list = list(np.argsort(values.values))
    ranked_list = values.indices[ranked_list]
    precision_recall = _curve_precision_recall_ranking(
        info[flipped_labels], ranked_list
    )
    return auc(precision_recall.index, precision_recall.values)


def metric_weighted_metric_drop(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    eval_model: SupervisedModel,
    metric: str = "accuracy",
    progress: bool = False,
) -> float:
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
        progress: Whether to display a progress bar.

    Returns:
        The weighted reciprocal difference average in the given metric.
    """
    u_eval = Utility(
        data=data,
        model=deepcopy(eval_model),
        scorer=Scorer(metric, default=0),
        catch_errors=True,
    )
    metrics = _curve_score_over_point_removal_or_addition(
        u_eval, values, highest_point_removal=True, progress=progress
    )
    diff_scores = metrics.diff(-1).values[:-1]
    diff_scores = np.nancumsum(diff_scores)
    weighted_diff_scores = diff_scores / (np.arange(1, len(diff_scores) + 1))
    return float(np.sum(weighted_diff_scores))


def curve_precision_recall_valuation_result(
    data: Dataset, values: ValuationResult, info: Dict, flipped_labels: str
) -> pd.Series:
    """
    Computes the precision-recall curve for a given valuation result on a dataset.

    Args:
        data: Dataset to compute the precision-recall curve on.
        values: Valuation result to compute the precision-recall curve for.
        info: Dictionary containing information about the dataset.
        flipped_labels: Name of the key in the info dictionary containing the indices of
            the flipped labels.

    Returns:
        A pd.Series object containing the precision and recall values for each prefix of
        the ranking. The index of the series is the recall and the values are the
        corresponding precision values.
    """
    ranked_list = list(np.argsort(values.values))
    ranked_list = values.indices[ranked_list]
    return _curve_precision_recall_ranking(info[flipped_labels], ranked_list)


def curve_score_over_point_removal_or_addition(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    eval_model: SupervisedModel,
    metric: str = "accuracy",
    highest_point_removal: bool = True,
    progress: bool = False,
) -> pd.Series:
    """
    Computes the utility score over the removal or addition of points. The utility score
    is computed for each prefix of the ranking. The prefix is either the highest or the
    lowest points of the ranking with respect to the valuation result.

    Args:
        data: Dataset to compute the utility score on.
        values: Valuation result to rank the data points.
        info: Additional information about the dataset.
        eval_model: Evaluation model to use for evaluation.
        metric: Scorer metric to use for evaluation.
        highest_point_removal: True, if the highest points should be removed. False, if
            the lowest points should be added.
        progress: Whether to display a progress bar.

    Returns:
        A pd.Series object containing the utility scores for each prefix of the ranking.
    """
    u_eval = Utility(
        data=data,
        model=deepcopy(eval_model),
        scorer=Scorer(metric, default=np.nan),
        catch_errors=True,
    )
    curve = _curve_score_over_point_removal_or_addition(
        u_eval,
        values,
        highest_point_removal=highest_point_removal,
        progress=progress,
    )
    curve.name = metric
    return curve


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
    min_points: int = 8,
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

    values.sort(reverse=highest_point_removal)
    scores = pd.Series(index=np.arange(len(u.data) - min_points + 1), dtype=np.float64)
    index = []

    for j in maybe_progress(
        len(u.data) - min_points + 1, display=progress, desc="Removal Scores"
    ):
        scores.iloc[j] = (
            u(values.indices[j:])
            if highest_point_removal
            else u(values.indices[: min_points + j])
        )
        index.append(j if highest_point_removal else j + min_points)

    scores.index = index
    scores.index.name = "n_points_" + ("removed" if highest_point_removal else "added")
    return scores
