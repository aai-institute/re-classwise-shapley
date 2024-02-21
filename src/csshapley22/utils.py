import logging
import os
import random

import numpy as np
import pandas as pd
import torch

__all__ = [
    "set_random_seed",
    "setup_logger",
    "compute_values",
    "convert_values_to_dataframe",
]

from pydvl.utils import Utility
from pydvl.value import ValuationResult

from csshapley22.algo.class_wise import class_wise_shapley


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
    elif method_name == "Class Wise":
        values = class_wise_shapley(utility)
    else:
        raise NotImplementedError

    return values


def convert_values_to_dataframe(values: ValuationResult) -> pd.DataFrame:
    df = (
        values.to_dataframe(column="value")
        .drop(columns=["value_stderr"])
        .T.reset_index(drop=True)
    )
    df = df[sorted(df.columns)]
    return df
