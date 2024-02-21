import io
from contextlib import redirect_stderr
from enum import Enum

from pydvl.utils import Utility
from pydvl.value import (
    MaxUpdates,
    RelativeTruncation,
    ShapleyMode,
    ValuationResult,
    compute_shapley_values,
    naive_loo,
)
from pydvl.value.shapley.classwise import class_wise_shapley


def compute_values(
    utility: Utility, valuation_method: str, **kwargs
) -> ValuationResult:
    if valuation_method == "random":
        values = ValuationResult.from_random(size=len(utility.data))
    elif valuation_method == "leave_one_out":
        values = naive_loo(utility, progress=False)
    elif valuation_method == "class_wise":
        # TODO: Talk about budget parameter in this context
        n_updates = 200  # budget // len(utility.data)
        kwargs = {
            "done": MaxUpdates(n_updates),
        }
        values = class_wise_shapley(utility, progress=True, **kwargs)

    elif valuation_method == "truncated_monte_carlo":
        mode = ShapleyMode.TruncatedMontecarlo
        # The budget for TMCShapley methods is less because
        # for each iteration it goes over all indices
        # of an entire permutation of indices
        n_updates = 200
        kwargs = {
            "truncation": RelativeTruncation(utility, rtol=0.01),
            "done": MaxUpdates(n_updates),
        }
        f = io.StringIO()
        with redirect_stderr(f):
            values = compute_shapley_values(
                utility,
                mode=mode,
                **kwargs,
            )
    else:
        raise NotImplementedError(
            f"The method {valuation_method} is not registered within."
        )

    return values
