import io
import logging
import os
import random
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import torch

__all__ = [
    "set_random_seed",
    "setup_logger",
    "compute_values",
    "convert_values_to_dataframe",
    "instantiate_model",
]

from pydvl.utils import Utility
from pydvl.value import MaxUpdates, ValuationResult, compute_shapley_values, naive_loo
from pydvl.value.shapley import ShapleyMode
from pydvl.value.shapley.classwise import class_wise_shapley
from pydvl.value.shapley.truncated import RelativeTruncation
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def set_random_seed(seed: int) -> None:
    """Taken verbatim from:
    https://koustuvsinha.com//practices_for_reproducibility/
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logger


def compute_values(
    method_name: str, utility: Utility, n_jobs: int, budget: int
) -> ValuationResult:
    if method_name == "Random":
        values = ValuationResult.from_random(size=len(utility.data))
    elif method_name == "Leave One Out":
        values = naive_loo(utility, progress=False)
    elif method_name == "Class Wise":
        # TODO: Talk about budget parameter in this context
        n_updates = 3  # budget // len(utility.data)
        kwargs = {
            "done": MaxUpdates(n_updates),
        }
        values = class_wise_shapley(utility, **kwargs)

    elif method_name == "TMC":
        mode = ShapleyMode.TruncatedMontecarlo
        # The budget for TMCShapley methods is less because
        # for each iteration it goes over all indices
        # of an entire permutation of indices
        n_updates = 3
        kwargs = {
            "truncation": RelativeTruncation(utility, rtol=0.01),
            "done": MaxUpdates(n_updates),
        }
        f = io.StringIO()
        with redirect_stderr(f):
            values = compute_shapley_values(
                utility,
                mode=mode,
                n_jobs=n_jobs,
                **kwargs,
            )
    else:
        raise NotImplementedError(f"The method {method_name} is not registered within.")

    return values


def convert_values_to_dataframe(values: ValuationResult) -> pd.DataFrame:
    df = (
        values.to_dataframe(column="value")
        .drop(columns=["value_stderr"])
        .T.reset_index(drop=True)
    )
    df = df[sorted(df.columns)]
    return df


def instantiate_model(model_name: str) -> Pipeline:
    # We do not set the random_state in the model itself
    # because we are testing the method and not the model
    if model_name == "GradientBoostingRegressor":
        model = make_pipeline(GradientBoostingClassifier())
    elif model_name == "LogisticRegression":
        model = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", n_jobs=1)
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")
    return model
