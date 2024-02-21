import json
import os
import pickle
from functools import partial
from typing import Callable, Dict, Literal

import click
import pandas as pd
from dvc.api import params_show
from pydvl.utils import Dataset
from pydvl.value import ValuationResult

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.metric import (
    curve_precision_recall_valuation_result,
    curve_score_over_point_removal_or_addition,
    metric_roc_auc,
    metric_weighted_metric_drop,
)
from re_classwise_shapley.model import instantiate_model

logger = setup_logger("evaluate_metrics")

# Type -> Metric -> Function
MetricRegistry = {
    "metric": {
        "weighted_metric_drop": metric_weighted_metric_drop,
        "roc_auc": metric_roc_auc,
    },
    "curve": {
        "point_removal": curve_score_over_point_removal_or_addition,
        "precision_recall": curve_precision_recall_valuation_result,
    },
}


def instantiate_metric_functions(
    type: Literal["metric", "curve"],
    idx: Literal["weighted_metric_drop", "roc_auc"],
    **kwargs,
) -> partial:
    if type not in MetricRegistry.keys():
        raise NotImplementedError(f"Type {type} is not registered.")

    type_metric_registry = MetricRegistry[type]
    if idx not in type_metric_registry.keys():
        raise NotImplementedError(f"Idx {idx} for type {type} is not registered.")

    return partial(type_metric_registry[idx], **kwargs)


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method", type=str, required=True)
@click.option("--metric-name", type=str, required=True)
def evaluate_metrics(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method: str,
    repetition_id: int,
    metric_name: str,
):
    """
    Calculate data values for a specified dataset.
    :param dataset_name: Dataset to use.
    """
    input_dir = (
        Accessor.VALUES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
    )
    os.makedirs(input_dir, exist_ok=True)
    with open(
        input_dir / f"valuation.{valuation_method}.pkl",
        "rb",
    ) as file:
        values = pickle.load(file)

    data_dir = (
        Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    with open(data_dir / "test_set.pkl", "rb") as file:
        test_set = pickle.load(file)

    preprocess_info_filename = data_dir / "preprocess_info.json"
    if os.path.exists(preprocess_info_filename):
        with open(data_dir / "preprocess_info.json", "r") as file:
            preprocess_info = json.load(file)
    else:
        preprocess_info = {}

    output_dir = (
        Accessor.RESULT_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
        / valuation_method
    )
    os.makedirs(output_dir, exist_ok=True)

    params = params_show()
    metrics = params["experiments"][experiment_name]["metrics"]
    metric_kwargs = metrics[metric_name]
    metric_type = metric_kwargs.pop("type")
    metric_idx = metric_kwargs.pop("idx")
    metric_fn = instantiate_metric_functions(metric_type, metric_idx, **metric_kwargs)
    metric_values = metric_fn(test_set, values, preprocess_info)
    file_name = output_dir / f"{metric_name}.csv"

    match metric_type:
        case "metric":
            evaluated_metrics = pd.Series([metric_values])
            evaluated_metrics.name = "value"
            evaluated_metrics.index.name = "metric"
            evaluated_metrics.to_csv(file_name)

        case "curve":
            metric_values.to_csv(file_name)


if __name__ == "__main__":
    evaluate_metrics()
