from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydvl.utils import Scorer, SupervisedModel, Utility, maybe_progress
from pydvl.value.result import ValuationResult
from scipy.stats import gaussian_kde
from sklearn.metrics import auc

from re_classwise_shapley.log import setup_logger

__all__ = [
    "pr_curve_ranking",
    "weighted_metric_drop",
    "weighted_reciprocal_diff_average",
]


logger = setup_logger(__name__)


def pr_curve_ranking(
    target_list: NDArray[np.int_], ranked_list: NDArray[np.int_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_], float]:
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

    return p, r, auc(r, p)


def roc_auc_pr_recall(
    test_utility: Utility, values: ValuationResult, info: Dict
) -> Tuple[float, pd.Series]:
    ranked_list = list(np.argsort(values))
    ranked_list = test_utility.data.indices[ranked_list]
    precision, recall, score = pr_curve_ranking(info["idx"], ranked_list)
    logger.debug("Computing precision-recall curve on separate test set..")
    graph = pd.Series(precision, index=recall)
    graph = graph[~graph.index.duplicated(keep="first")]
    graph = graph.sort_index(ascending=True)
    graph.index.name = "recall"
    graph.name = "precision"
    return score, graph


def weighted_reciprocal_diff_average(
    u: Utility,
    values: ValuationResult,
    *,
    progress: bool = False,
    highest_first: bool = True,
) -> Tuple[float, pd.Series]:
    """
    Calculates the weighted reciprocal difference average of a valuation function.
    :param u: :class:`~pydvl.utils.utility.Utility` object holding data, model
        and scoring function.
    :param values: :class:`~pydvl.value.result.ValuationResult` object holding the
        values of the dataset.
    :param progress: Whether to display a progress bar.
    :param highest_first: Whether to sort the datapoints by their values in descending
        order.
    :return: A tuple containing the average and a :class:`pandas.Series` object
        containing the scores for each step.
    """
    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values)}, should be equal to the number of"
            f" data indices, {len(u.data.indices)}"
        )

    values.sort(reverse=highest_first)
    scores = pd.Series(index=np.arange(len(u.data)), dtype=np.float64)
    scores.iloc[0] = u(u.data.indices) if highest_first else u(set())

    for j in maybe_progress(len(u.data) - 1, display=progress, desc="Removal Scores"):
        scores.iloc[j + 1] = (
            u(values.indices[j + 1 :]) if highest_first else u(values.indices[: j + 1])
        )

    diff_scores = scores.diff(-1).values[:-1]
    diff_scores = np.nancumsum(diff_scores)
    weighted_diff_scores = diff_scores / (np.arange(1, len(diff_scores) + 1))
    avg = np.sum(weighted_diff_scores)
    if not highest_first:
        avg *= -1

    return float(avg), scores


def weighted_metric_drop(
    u: Utility,
    values: ValuationResult,
    info: Dict,
    *,
    progress: bool = False,
    highest_first: bool = True,
    transfer_model: SupervisedModel = None,
    metric: str = "accuracy",
) -> Tuple[float, pd.Series]:
    u_eval = Utility(
        data=u.data,
        model=deepcopy(transfer_model),
        scorer=Scorer(metric, default=0),
    )
    wad, graph = weighted_reciprocal_diff_average(
        u=u_eval,
        values=values,
        progress=progress,
        highest_first=highest_first,
    )
    graph.index.name = "num_removed"
    graph.name = metric
    return float(wad), graph
