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

    for dataset_name in active_params["datasets"]:
        logger.info(f"Fetching dataset {dataset_name}.")
        _fetch_data(dataset_name)
        logger.info(f"Preprocessing dataset {dataset_name}.")
        _preprocess_data(dataset_name)

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
        logger.info(
            f"Sample dataset {dataset_name} for experiment {experiment_name} and "
            f"seed {repetition_id}."
        )
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
        logger.info(f"Running experiment {experiment_name} with model {model_name}.")
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
            logger.info(
                f"Calculate values for dataset {dataset_name}, valuation method "
                f"{valuation_method_name} and seed {repetition_id}."
            )
            _calculate_values(
                experiment_name,
                dataset_name,
                model_name,
                valuation_method_name,
                repetition_id,
            )

            for metric_name in params["experiments"][experiment_name]["metrics"].keys():
                logger.info(
                    f"Calculate metric {metric_name} for dataset {dataset_name}, "
                    f"valuation method {valuation_method_name} and seed "
                    f"{repetition_id}."
                )
                logger.info(f"Evaluate metric {metric_name}.")
                _evaluate_metrics(
                    experiment_name,
                    dataset_name,
                    model_name,
                    valuation_method_name,
                    repetition_id,
                    metric_name,
                )

        logger.info(f"Render plots for {experiment_name} and {model_name}.")
        _render_plots(experiment_name, model_name)


if __name__ == "__main__":
    run_pipeline()