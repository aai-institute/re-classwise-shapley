import functools
import multiprocessing.pool
import os
import random
import subprocess
import time
from typing import Callable, Optional, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd
import torch

__all__ = [
    "set_random_seed",
    "convert_values_to_dataframe",
    "instantiate_model",
    "timeout",
]

from pydvl.value import ValuationResult
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from csshapley22.log import setup_logger

logger = setup_logger()


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


def convert_values_to_dataframe(values: ValuationResult) -> pd.DataFrame:
    df = (
        values.to_dataframe(column="value")
        .drop(columns=["value_stderr"])
        .T.reset_index(drop=True)
    )
    df = df[sorted(df.columns)]
    return df


def instantiate_model(model_name: str, **model_kwargs) -> Pipeline:
    if model_name == "gradient_boosting_classifier":
        model = make_pipeline(GradientBoostingClassifier(**model_kwargs))
    elif model_name == "logistic_regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(**model_kwargs))
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return model


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def timeout(max_timeout: int):
    """Timeout decorator, parameter in seconds."""

    processing_params = ParamSpec("processing_params")
    processing_result = TypeVar("processing_result")

    def timeout_decorator(fn: Callable[processing_params, processing_result]):
        @functools.wraps(fn)
        def func_wrapper(
            *args: processing_params.args, **kwargs: processing_params.kwargs
        ) -> Optional[processing_result]:
            """Closure for function."""

            start_time = time.time()
            pool = multiprocessing.pool.ThreadPool(processes=1)
            logger.info(f"Thread pool created with 1 process for executing {fn}.")
            async_result = pool.apply_async(fn, cast(list, args), cast(dict, kwargs))
            logger.info(
                f"Called async function {fn} with arguments {args} and kwargs {kwargs}."
            )

            try:
                result = async_result.get(max_timeout)
            except multiprocessing.context.TimeoutError:
                logger.error(f"Function call timed out after {max_timeout} seconds.")
                pool.terminate()
                return None

            pool.terminate()
            passed_time = time.time() - start_time
            logger.info(f"Function call took {passed_time} seconds.")
            return result

        return func_wrapper

    return timeout_decorator
