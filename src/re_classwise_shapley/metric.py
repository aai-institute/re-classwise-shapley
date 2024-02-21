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
    "pr_curve_ranking",
    "weighted_metric_drop",
]


logger = setup_logger(__name__)


def pr_curve_ranking(
    target_list: NDArray[np.int_], ranked_list: NDArray[np.int_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Calculates the precision-recall curve for a given target list and ranked list.
    Also calculates the area under the curve (AUC).
    :param target_list: The list of target indices. A subset of the ranked list.
    :param ranked_list: The list of ranked indices.
    :return: Tuple of precision, recall and AUC.
    """
    p, r = np.zeros(len(ranked_list)), np.zeros(len(ranked_list))
    for idx in range(len(ranked_list)):
        partial_list = ranked_list[: idx + 1]
        intersection = list(set(target_list) & set(partial_list))
        p[idx] = float(len(intersection) / len(partial_list))
        r[idx] = float(len(intersection) / len(target_list))

    return p, r


def roc_auc(
    data: Dataset, values: ValuationResult, info: Dict, flipped_labels: str
) -> float:
    ranked_list = list(np.argsort(values))
    ranked_list = data.indices[ranked_list]
    precision, recall = pr_curve_ranking(info[flipped_labels], ranked_list)
    return auc(recall, precision)


def pr_recall(
    data: Dataset, values: ValuationResult, info: Dict, flipped_labels: str
) -> pd.Series:
    ranked_list = list(np.argsort(values))
    ranked_list = data.indices[ranked_list]
    precision, recall = pr_curve_ranking(info[flipped_labels], ranked_list)
    logger.debug("Computing precision-recall curve on separate test set..")
    graph = pd.Series(precision, index=recall)
    graph = graph[~graph.index.duplicated(keep="first")]
    graph = graph.sort_index(ascending=True)
    graph.index.name = "recall"
    graph.name = "precision"
    return graph


def accuracies_point_removal_or_addition(
    u: Utility,
    values: ValuationResult,
    info: Dict,
    *,
    progress: bool = False,
    highest_point_removal: bool = False,
) -> pd.Series:
    """
    Calculates the weighted reciprocal difference average of a valuation function.
    :param u: :class:`~pydvl.utils.utility.Utility` object holding data, model
        and scoring function.
    :param values: :class:`~pydvl.value.result.ValuationResult` object holding the
        values of the dataset.
    :param progress: Whether to display a progress bar.
    :param highest_point_removal: Whether to sort the datapoints by their values in descending
        order.
    :return: A tuple containing the average and a :class:`pandas.Series` object
        containing the scores for each step.
    """
    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values)}, should be equal to the number of"
            f" data indices, {len(u.data.indices)}"
        )

    min_points = 8
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


def weighted_metric_drop(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    eval_model: SupervisedModel,
    metric: str = "accuracy",
    progress: bool = False,
) -> float:
    u_eval = Utility(
        data=data,
        model=deepcopy(eval_model),
        scorer=Scorer(metric, default=0),
        catch_errors=True,
    )
    metrics = accuracies_point_removal_or_addition(
        u_eval, values, info, highest_point_removal=True, progress=progress
    )
    diff_scores = metrics.diff(-1).values[:-1]
    diff_scores = np.nancumsum(diff_scores)
    weighted_diff_scores = diff_scores / (np.arange(1, len(diff_scores) + 1))
    avg = np.sum(weighted_diff_scores)
    return float(avg)


def metric_curve(
    data: Dataset,
    values: ValuationResult,
    info: Dict,
    eval_model: SupervisedModel,
    metric: str = "accuracy",
    progress: bool = False,
    highest_point_removal: bool = True,
) -> pd.Series:
    u_eval = Utility(
        data=data,
        model=deepcopy(eval_model),
        scorer=Scorer(metric, default=np.nan),
        catch_errors=True,
    )
    curve = accuracies_point_removal_or_addition(
        u_eval,
        values,
        info,
        highest_point_removal=highest_point_removal,
        progress=progress,
    )
    curve.name = metric
    return curve
