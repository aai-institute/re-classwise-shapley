"""
Stage five evaluates metrics using calculated Shapley values.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Evaluate metrics using calculated Shapley values as specified in the
`params.experiments` section. All files are stored in the `Accessor.RESULT_PATH`
directory. The metrics are usually stored as `*.csv` files. Each metric consists of
a single value and a curve. The curve is stored as `*.curve.csv` file.
"""
import os
from functools import partial

import click
import pandas as pd
from pydvl.utils.functional import free_arguments, maybe_add_argument

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.metric import MetricsRegistry
from re_classwise_shapley.requests import FunctionalCurveRequest
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("evaluate_metrics")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method-name", type=str, required=True)
@click.option("--metric-name", type=str, required=True)
def evaluate_metrics(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
    metric_name: str,
):
    """
    Evaluate a curve on the calculated values. The curve is specified in the
    `params.experiments` section. The values are stored in the `Accessor.RESULT_PATH`.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
        model_name: Model to use. As specified in the `params.models` section.
        valuation_method_name: Name of the valuation method to use. As specified in the
            `params.valuation_methods` section.
        repetition_id: Repetition id of the experiment. It is used also as a seed for
            all randomness.
        metric_name: Name of the metric to use. As specified in the `metrics` section of
            the current experiment in `params.experiment` section.
    """
    _evaluate_metrics(
        experiment_name,
        dataset_name,
        model_name,
        valuation_method_name,
        repetition_id,
        metric_name,
    )


def _evaluate_metrics(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
    metric_name: str,
):
    logger.info(
        f"Evaluate metrics for {experiment_name}/{dataset_name}/{model_name}/{valuation_method_name}/{repetition_id}/{metric_name}."
    )
    output_dir = (
        Accessor.METRICS_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
        / valuation_method_name
    )

    params = load_params_fast()
    metrics = params["experiments"][experiment_name]["metrics"]
    metrics_kwargs = metrics[metric_name]
    metric_fn = metrics_kwargs.pop("fn")
    metrics_kwargs.pop("plots", None)
    curve_names = metrics_kwargs.pop("curve")
    repetitions = get_active_repetitions(params)

    fn = MetricsRegistry[metric_fn]
    requests = []
    if "random_base_line" in free_arguments(fn):
        requests.append(
            FunctionalCurveRequest("random_base_line", "random", repetitions)
        )

    fn = maybe_add_argument(fn, "random_base_line")
    metrics_fn = partial(fn, **metrics_kwargs)

    os.makedirs(output_dir, exist_ok=True)
    curves = list(
        Accessor.curves(
            experiment_name,
            model_name,
            dataset_name,
            valuation_method_name,
            curve_names,
            repetition_id,
        ).iterrows()
    )
    for i, (_, curve) in enumerate(curves):
        curve_name = curve["curve_name"]
        logger.info(f"Processing curve {i+1}/{len(curves)} with name '{curve_name}'.")
        curve = curve["curve"]
        if os.path.exists(output_dir / f"{metric_name}.{curve_name}.csv"):
            logger.info(
                f"Metric data exists in '{output_dir}/{metric_name}.{curve_name}.csv'. Skipping..."
            )
            continue

        extra_kwargs = {
            request.arg_name: request.request(
                experiment_name, model_name, dataset_name, curve_names
            )
            for request in requests
        }
        metric = metrics_fn(curve, **extra_kwargs)
        evaluated_metrics = pd.Series([metric])
        evaluated_metrics.name = "value"
        evaluated_metrics.index.name = "metric"
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        evaluated_metrics.to_csv(metric_dir / f"{curve_name}.csv")


def get_active_repetitions(params):
    active_params = params["active"]
    repetitions = active_params["repetitions"]
    return list(repetitions)


if __name__ == "__main__":
    evaluate_metrics()
