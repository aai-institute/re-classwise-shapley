import logging
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch

__all__ = [
    "set_random_seed",
    "setup_logger",
    "convert_values_to_dataframe",
    "instantiate_model",
]

from pydvl.value import ValuationResult
from sklearn.ensemble import GradientBoostingClassifier
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
