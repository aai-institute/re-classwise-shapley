from typing import Dict, Iterable, Union

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Utility, maybe_progress
from pydvl.value.result import ValuationResult

__all__ = ["weighted_reciprocal_diff_average"]

from sklearn.metrics import accuracy_score


def weighted_reciprocal_diff_average(
    u: Utility,
    values: ValuationResult,
    *,
    progress: bool = False,
) -> float:
    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values) }, should be equal to the number of data indices, {len(u.data.indices)}"
        )

    # We sort in descending order if we want to remove the best values
    values.sort(reverse=True)
    full_accuracy = u(u.data.indices)
    avg = 0

    for j in maybe_progress(len(u.data), display=progress, desc="Removal Scores"):
        j_accuracy = u(u.data.indices[j + 1 :])
        discount_factor = 1 / (j + 1)
        new_term = discount_factor * (full_accuracy - j_accuracy)

        if not np.isnan(new_term):
            avg += new_term

    return avg
