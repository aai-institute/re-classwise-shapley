from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydvl.utils import Utility, maybe_progress
from pydvl.value.result import ValuationResult
from sklearn.metrics import auc

__all__ = ["pr_curve_ranking", "weighted_reciprocal_diff_average"]


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
    p, r = np.empty(len(ranked_list)), np.empty(len(ranked_list))
    for idx in range(len(ranked_list)):
        partial_list = ranked_list[: idx + 1]
        intersection = list(set(target_list) & set(partial_list))
        r[idx] = float(len(intersection) / len(target_list))
        p[idx] = float(len(intersection) / len(partial_list))

    return r, p, auc(r, p)


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
    scores = pd.Series(index=np.arange(len(u.data)) + 1, dtype=np.float64)
    scores.loc[0] = u(u.data.indices)

    for j in maybe_progress(len(u.data), display=progress, desc="Removal Scores"):
        scores.loc[j + 1] = u(values.indices[j + 1 :])

    diff_scores = scores.diff(-1).values[:-1]
    diff_scores = np.nancumsum(diff_scores)
    weighted_diff_scores = diff_scores / (np.arange(1, len(diff_scores) + 1))
    avg = np.sum(weighted_diff_scores)
    return float(avg), scores
