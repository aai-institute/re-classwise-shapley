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
from numpy._typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

__all__ = [
    "set_random_seed",
    "convert_values_to_dataframe",
    "instantiate_model",
    "timeout",
]

from pydvl.utils import SupervisedModel
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


def instantiate_model(model_name: str, **model_kwargs) -> SupervisedModel:
    if model_name == "gradient_boosting_classifier":
        model = make_pipeline(GradientBoostingClassifier(**model_kwargs))
    elif model_name == "logistic_regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(**model_kwargs))
    elif model_name == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(**model_kwargs))
    elif model_name == "svm":
        model = make_pipeline(StandardScaler(), SVC(**model_kwargs))
    elif model_name == "mlp":
        model = make_pipeline(StandardScaler(), MLPClassifier(**model_kwargs))
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return WrapperModel(model)


class WrapperModel:
    def __init__(self, model: SupervisedModel):
        self.model = model
        self._unique_cls = None

    def fit(self, x: NDArray[np.float_], y: NDArray[np.int_]):
        if len(np.unique(y)) == 1:
            self._unique_cls = y[0]
        else:
            self._unique_cls = None
            self.model.fit(x, y)

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.int_]:
        if self._unique_cls is None:
            return self.model.predict(x)
        else:
            return np.ones(len(x), dtype=int) * self._unique_cls

    def score(self, x: NDArray[np.float_], y: NDArray[np.int_]) -> float:
        return self.model.score(x, y)


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


def order_dict(dictionary):
    return {
        k: order_dict(v) if isinstance(v, dict) else v
        for k, v in sorted(dictionary.items())
    }
