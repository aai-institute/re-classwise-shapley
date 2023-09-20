import json
import os
import pickle
from enum import Enum
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


def instantiate_metric_functions(
    prefix: str,
    idx: Literal["weighted_metric_drop", "roc_auc"],
    **kwargs,
) -> Dict[str, Callable[[Dataset, ValuationResult, Dict], float]]:
    """
    Dispatches the metrics to functions accepting a dataset, valuation result and
    preprocess info.

    Args:
        prefix: Prefix for the keys to be used to identify the metrics.
        idx: Identifier for the metric to be used.
        **kwargs: Contains kw arguments for metrics.

    Returns:
        A dictionary of functions to calculate the metrics.
    """
    params = params_show()
    metrics = {}
    if idx == "weighted_metric_drop":
        eval_models = kwargs["eval_models"]
        metric = kwargs["metric"]
        for eval_model in eval_models:
            metric_key = f"{prefix}.{eval_model}"
            transfer_model_kwargs = params["models"][eval_model]
            eval_model_instance = instantiate_model(eval_model, transfer_model_kwargs)
            metrics[metric_key] = partial(
                metric_weighted_metric_drop,
                eval_model=eval_model_instance,
                metric=metric,
            )

        return metrics  # type: ignore
    elif idx == "roc_auc":
        metrics[prefix] = partial(
            metric_roc_auc, flipped_labels=kwargs["flipped_labels"]
        )
        return metrics  # type: ignore

    else:
        raise NotImplementedError(f"Metric {idx} is not implemented.")


def instantiate_curves(
    prefix: str,
    idx: str,
    **kwargs,
) -> Dict[str, Callable[[Dataset, ValuationResult, Dict], pd.DataFrame]]:
    """
    Dispatches the metrics to functions accepting a dataset, valuation result and
    preprocess info.

    Args:
        prefix: Prefix for the keys to be used to identify the metrics.
        idx: Identifier for the metric to be used.
        **kwargs: Contains kw arguments for metrics.

    Returns:
        A dictionary of functions to calculate the curves.
    """
    params = params_show()
    curves = {}
    if idx == "highest_point_removal" or idx == "lowest_point_addition":
        highest_point_removal = idx == "highest_point_removal"
        metric = kwargs["metric"]
        eval_models = kwargs["eval_models"]
        for eval_model in eval_models:
            curve_key = f"{prefix}.{eval_model}"
            transfer_model_kwargs = params["models"][eval_model]
            transfer_model = instantiate_model(eval_model, transfer_model_kwargs)
            curves[curve_key] = partial(
                curve_score_over_point_removal_or_addition,
                eval_model=transfer_model,
                metric=metric,
                highest_point_removal=highest_point_removal,
            )

        return curves  # type: ignore
    elif idx == "precision_recall":
        flipped_labels = kwargs["flipped_labels"]
        curves[prefix] = partial(
            curve_precision_recall_valuation_result, flipped_labels=flipped_labels
        )
        return curves  # type: ignore
    else:
        raise NotImplementedError(f"Curve {idx} is not implemented.")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method", type=str, required=True)
def evaluate_metrics(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method: str,
    repetition_id: int,
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

    eval_metrics = {}
    for metric_name, metric_kwargs in metrics.items():
        eval_metrics.update(instantiate_metric_functions(metric_name, **metric_kwargs))

    evaluated_metrics = {}
    for eval_metric_key, eval_metric in eval_metrics.items():
        logger.info("Evaluating metric %s", eval_metric_key)
        eval_metric_value = eval_metric(test_set, values, preprocess_info)
        evaluated_metrics[eval_metric_key] = eval_metric_value

    evaluated_metrics = pd.Series(evaluated_metrics)
    evaluated_metrics.name = "value"
    evaluated_metrics.index.name = "metric"
    evaluated_metrics.to_csv(output_dir / f"metrics.csv")

    curves = params["experiments"][experiment_name]["curves"]
    eval_curves = {}
    for curve_name, curve_kwargs in curves.items():
        eval_curves.update(instantiate_curves(curve_name, **curve_kwargs))

    for eval_curve_key, eval_curve in eval_curves.items():
        logger.info("Evaluating curve %s", eval_curve_key)
        eval_curve_value = eval_curve(test_set, values, preprocess_info)
        file_name = output_dir / f"{eval_curve_key}.csv"
        eval_curve_value.to_csv(file_name)


if __name__ == "__main__":
    evaluate_metrics()
