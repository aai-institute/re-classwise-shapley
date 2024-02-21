import io
import logging
from contextlib import redirect_stderr

from pydvl.utils import ParallelConfig, Utility
from pydvl.value import (
    MaxUpdates,
    RelativeTruncation,
    ShapleyMode,
    ValuationResult,
    compute_shapley_values,
    naive_loo,
)
from pydvl.value.semivalues import SemiValueMode, compute_semivalues
from pydvl.value.shapley.classwise import _class_wise_shapley_worker, class_wise_shapley


def compute_values(
    utility: Utility, valuation_method: str, **kwargs
) -> ValuationResult:
    progress = kwargs.get("progress", False)
    if valuation_method == "random":
        values = ValuationResult.from_random(size=len(utility.data))

    elif valuation_method == "loo":
        values = naive_loo(utility, progress=progress)

    elif valuation_method == "cs_shapley":
        values = _class_wise_shapley_worker(
            utility.data.indices,
            utility,
            progress=progress,
            done=MaxUpdates(kwargs["n_updates"]),
            truncation=RelativeTruncation(utility, rtol=1e-5),
        )

    elif valuation_method == "beta_shapley":
        values = compute_semivalues(
            u=utility,
            mode=SemiValueMode.BetaShapley,
            done=MaxUpdates(kwargs["n_updates"]),
            alpha=kwargs["alpha"],
            beta=kwargs["beta"],
            progress=progress,
        )

    elif valuation_method == "tmc_shapley":
        f = io.StringIO()
        with redirect_stderr(f):
            values = compute_shapley_values(
                utility,
                mode=ShapleyMode.TruncatedMontecarlo,
                truncation=RelativeTruncation(utility, rtol=kwargs["rtol"]),
                done=MaxUpdates(kwargs["n_updates"]),
                progress=progress,
            )

    else:
        raise NotImplementedError(
            f"The method {valuation_method} is not registered within."
        )

    return values
