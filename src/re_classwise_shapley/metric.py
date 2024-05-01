import math as m
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from re_classwise_shapley.log import setup_logger

__all__ = ["MetricsRegistry"]

logger = setup_logger(__name__)


def metric_roc_auc(
    precision_recall: pd.Series,
) -> float:
    return auc(precision_recall.index, precision_recall.values)


def metric_geometric_weighted_metric_drop(curve: pd.Series, input_perc: float) -> float:
    if input_perc < 1:
        n = m.ceil(len(curve) / 2)
        curve = curve.iloc[:n]

    diff_curve = curve.diff(-1).values[:-1]
    diff_curve = np.nancumsum(diff_curve)
    weighted_diff = diff_curve / (np.arange(1, len(diff_curve) + 1))
    return float(np.sum(weighted_diff))


def metric_weighted_relative_accuracy_difference_random(
    curve: pd.Series, lamb: float, random_base_line: pd.Series
) -> float:
    n = len(curve)
    weights = np.exp(-lamb * np.arange(n))
    return weights.dot(random_base_line - curve).mean()


MetricsRegistry: Dict[str, Callable[..., float]] = {
    "geometric_weighted_drop": metric_geometric_weighted_metric_drop,
    "roc_auc": metric_roc_auc,
    "weighted_relative_accuracy_difference_random": metric_weighted_relative_accuracy_difference_random,
}
