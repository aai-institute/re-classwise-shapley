from typing import Tuple

import numpy as np
import pandas as pd
from pydvl.utils import Utility, maybe_progress
from pydvl.value.result import ValuationResult

__all__ = ["weighted_reciprocal_diff_average"]


def weighted_reciprocal_diff_average(
    u: Utility,
    values: ValuationResult,
    *,
    progress: bool = False,
    highest_first: bool = True,
) -> Tuple[float, pd.Series]:
    if len(values) != len(u.data.indices):
        raise ValueError(
            f"The number of values, {len(values)}, should be equal to the number of"
            f" data indices, {len(u.data.indices)}"
        )

    values.sort(reverse=highest_first)
    full_accuracy = u(u.data.indices)
    avg = 0
    line = pd.Series(index=np.arange(len(u.data)), dtype=np.float64)

    for j in maybe_progress(len(u.data), display=progress, desc="Removal Scores"):
        j_accuracy = u(values.indices[j + 1 :])
        line.loc[j] = j_accuracy
        discount_factor = 1 / (j + 1)
        new_term = discount_factor * (full_accuracy - j_accuracy)

        if not np.isnan(new_term):
            avg += new_term

    return avg, line
