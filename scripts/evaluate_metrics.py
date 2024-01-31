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
import logging
import os
from functools import partial, reduce

import click
import pandas as pd
from pydvl.parallel import ParallelConfig
from pydvl.utils.functional import maybe_add_argument

from re_classwise_shapley.cache import PrefixMemcachedCacheBackend
from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.metric import MetricRegistry
from re_classwise_shapley.utils import load_params_fast, n_threaded, pipeline_seed

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
    Evaluate one metric on the calculated values. The metric is specified in the
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
    logger.info("Loading values, test set and preprocess info.")
    output_dir = (
        Accessor.RESULT_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
        / valuation_method_name
    )
    if os.path.exists(output_dir / f"{metric_name}.csv") and os.path.exists(
        output_dir / f"{metric_name}.curve.csv"
    ):
        return logger.info(f"Metric data exists in '{output_dir}'. Skipping...")

    values = Accessor.valuation_results(
        experiment_name, model_name, dataset_name, repetition_id, valuation_method_name
    ).loc[0, "valuation"]
    dataset = Accessor.datasets(experiment_name, dataset_name).loc[0]
    preprocess_info = dataset["preprocess_info"]

    os.makedirs(output_dir, exist_ok=True)

    params = load_params_fast()
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]
    parallel_config = ParallelConfig(
        backend=backend,
        n_cpus_local=n_jobs,
        logging_level=logging.INFO,
    )
    metrics = params["experiments"][experiment_name]["metrics"]
    metric_kwargs = metrics[metric_name]
    metric_idx = metric_kwargs.pop("idx")
    metric_kwargs.pop("len_curve_perc", None)
    metric_fn = partial(MetricRegistry[metric_idx], **metric_kwargs)
    metric_fn = reduce(
        maybe_add_argument,
        ["data", "values", "info", "n_jobs", "config", "progress", "seed", "cache"],
        metric_fn,
    )

    n_pipeline_step = 5
    seed = pipeline_seed(repetition_id, n_pipeline_step)
    cache = None
    if (
        "eval_model" in metric_kwargs
        and "cache_group" in params["valuation_methods"][valuation_method_name]
    ):
        cache_group = params["valuation_methods"][valuation_method_name]["cache_group"]
        prefix = f"{experiment_name}/{dataset_name}/{model_name}/{cache_group}"
        try:
            cache = PrefixMemcachedCacheBackend(prefix=prefix)
        except ConnectionRefusedError:
            logger.info("Couldn't connect to cache backend.")
            cache = None

    logger.info("Evaluating metric...")
    with n_threaded(n_threads=1):
        metric_values, metric_curve = metric_fn(
            data=dataset["test_set"],
            values=values,
            info=preprocess_info,
            n_jobs=n_jobs,
            config=parallel_config,
            progress=True,
            seed=seed,
            cache=cache,
        )

    evaluated_metrics = pd.Series([metric_values])
    evaluated_metrics.name = "value"
    evaluated_metrics.index.name = "metric"
    evaluated_metrics.to_csv(output_dir / f"{metric_name}.csv")
    metric_curve.to_csv(output_dir / f"{metric_name}.curve.csv")


if __name__ == "__main__":
    evaluate_metrics()
