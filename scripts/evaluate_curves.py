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
from pydvl.parallel import ParallelConfig
from pydvl.utils.functional import maybe_add_argument

from re_classwise_shapley.cache import PrefixMemcachedCacheBackend
from re_classwise_shapley.curve import CurvesRegistry
from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import load_params_fast, n_threaded, pipeline_seed

logger = setup_logger("evaluate_curves")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method-name", type=str, required=True)
@click.option("--curve-name", type=str, required=True)
def evaluate_curves(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
    curve_name: str,
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
        curve_name: Name of the metric to use. As specified in the `metrics` section of
            the current experiment in `params.experiment` section.
    """
    _evaluate_curves(
        experiment_name,
        dataset_name,
        model_name,
        valuation_method_name,
        repetition_id,
        curve_name,
    )


def _evaluate_curves(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
    curve_name: str,
):
    logger.info("Loading values, test set and preprocess info.")
    output_dir = (
        Accessor.CURVES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
        / valuation_method_name
    )
    if os.path.exists(output_dir / f"{curve_name}.csv"):
        return logger.info(f"Curve data exists in '{output_dir}'. Skipping...")

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
    curves = params["experiments"][experiment_name]["curves"]
    curves_kwargs = curves[curve_name]
    curves_idx = curves_kwargs.pop("fn")
    curves_kwargs.pop("plots", None)
    curves_fn = partial(CurvesRegistry[curves_idx], **curves_kwargs)
    curves_fn = reduce(
        maybe_add_argument,
        ["data", "values", "info", "n_jobs", "config", "progress", "seed", "cache"],
        curves_fn,
    )

    n_pipeline_step = 5
    seed = pipeline_seed(repetition_id, n_pipeline_step)
    cache = None
    if (
        "eval_model" in curves_kwargs
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
        metric_curve = curves_fn(
            data=dataset["test_set"],
            values=values,
            info=preprocess_info,
            n_jobs=n_jobs,
            config=parallel_config,
            progress=True,
            seed=seed,
            cache=cache,
        )

    metric_curve.to_csv(output_dir / f"{curve_name}.csv")


if __name__ == "__main__":
    evaluate_curves()
