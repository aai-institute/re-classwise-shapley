"""
Runs the whole pipeline without dvc.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots
"""
from itertools import product

import click
from calculate_values import _calculate_values
from evaluate_metrics import _evaluate_metrics
from fetch_data import _fetch_data
from preprocess_data import _preprocess_data
from render_plots import _render_plots
from sample_data import _sample_data

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("run_pipeline")


@click.command()
def run_pipeline():
    """
    Runs the whole pipeline without dvc.
    """
    params = load_params_fast()
    active_params = params["active"]

    logger.info("Fetching and preprocess datasets.")
    for dataset_name in active_params["datasets"]:
        _fetch_data(dataset_name)
        _preprocess_data(dataset_name)

    logger.info("Sample datasets.")
    for (
        experiment_name,
        dataset_name,
        repetition_id,
    ) in product(
        *[
            active_params[k]
            for k in [
                "experiments",
                "datasets",
                "repetitions",
            ]
        ]
    ):
        _sample_data(experiment_name, dataset_name, repetition_id)

    for experiment_name, model_name in product(
        *[
            active_params[k]
            for k in [
                "experiments",
                "models",
            ]
        ]
    ):
        logger.info(f"Calculate values for {experiment_name} and {model_name}.")
        for (
            dataset_name,
            valuation_method_name,
            repetition_id,
        ) in product(
            *[
                active_params[k]
                for k in [
                    "datasets",
                    "valuation_methods",
                    "repetitions",
                ]
            ]
        ):
            _calculate_values(
                experiment_name,
                dataset_name,
                model_name,
                valuation_method_name,
                repetition_id,
            )

            for metric in params["experiments"][experiment_name]["metrics"].keys():
                logger.info(f"Evaluate metric {metric}.")
                _evaluate_metrics(
                    experiment_name,
                    dataset_name,
                    model_name,
                    valuation_method_name,
                    repetition_id,
                    metric,
                )

        logger.info(f"Render plots values for {experiment_name} and {model_name}.")
        _render_plots(experiment_name, model_name)


if __name__ == "__main__":
    run_pipeline()
