"""
Runs the whole pipeline without dvc.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate Curves
6. Render plots
"""
import os
from itertools import product
from time import sleep

import click
from calculate_threshold_characteristics import _calculate_threshold_characteristics
from calculate_values import _calculate_values
from evaluate_curves import _evaluate_curves
from evaluate_metrics import _evaluate_metrics
from fetch_data import _fetch_data
from preprocess_data import _preprocess_data
from render_plots import _render_plots
from sample_data import _sample_data

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("run_pipeline")


def repeat(fn, *args, n_repeats: int = 3, sleep_seconds: float = 60, **kwargs):
    for n_try in range(n_repeats):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            logger.error(f"An error occurred with message {str(e)}")
            sleep(sleep_seconds)


@click.command()
def run_pipeline():
    """
    Runs the whole pipeline without dvc.
    """

    try:
        params = load_params_fast()
        stages = params["settings"]["stages"]
        active_params = params["active"]
        repetitions = active_params["repetitions"]
        active_params["repetitions"] = list(
            range(repetitions["from"], repetitions["to"] + 1)
        )

        for dataset_name in active_params["datasets"]:
            logger.info(f"Fetching dataset {dataset_name}.")
            if stages["fetch_data"]:
                repeat(_fetch_data, dataset_name)
            logger.info(f"Preprocessing dataset {dataset_name}.")
            if stages["preprocess_data"]:
                _preprocess_data(dataset_name)

        for (
            experiment_name,
            dataset_name,
        ) in product(
            *[
                active_params[k]
                for k in [
                    "experiments",
                    "datasets",
                ]
            ]
        ):
            logger.info(
                f"Sample dataset {dataset_name} for experiment {experiment_name}."
            )
            if stages["sample_data"]:
                _sample_data(experiment_name, dataset_name)

            if stages["calculate_threshold_characteristics"]:
                for repetition_id in active_params["repetitions"]:
                    repeat(
                        _calculate_threshold_characteristics,
                        experiment_name,
                        dataset_name,
                        repetition_id,
                    )

        for experiment_name, model_name in product(
            *[
                active_params[k]
                for k in [
                    "experiments",
                    "models",
                ]
            ]
        ):
            if stages["calculate_values"]:
                logger.info(
                    f"Running experiment {experiment_name} with model {model_name}."
                )
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
                    repeat(
                        _calculate_values,
                        experiment_name,
                        dataset_name,
                        model_name,
                        valuation_method_name,
                        repetition_id,
                    )

            for (
                dataset_name,
                valuation_method_name,
                repetition_id,
            ) in product(
                active_params["datasets"],
                active_params["valuation_methods"],
                active_params["repetitions"],
            ):
                if stages["evaluate_curves"]:
                    for curve_name in params["experiments"][experiment_name][
                        "curves"
                    ].keys():
                        logger.info(
                            f"Calculate metric {curve_name} for dataset {dataset_name}, "
                            f"valuation method {valuation_method_name} and seed "
                            f"{repetition_id}."
                        )
                        logger.info(f"Evaluate metric {curve_name}.")
                        repeat(
                            _evaluate_curves,
                            experiment_name,
                            dataset_name,
                            model_name,
                            valuation_method_name,
                            repetition_id,
                            curve_name,
                        )

                if stages["evaluate_metrics"]:
                    for metric_name in params["experiments"][experiment_name][
                        "metrics"
                    ].keys():
                        logger.info(
                            f"Calculate metric {metric_name} for dataset {dataset_name}, "
                            f"valuation method {valuation_method_name} and seed "
                            f"{repetition_id}."
                        )
                        logger.info(f"Evaluate metric {metric_name}.")
                        repeat(
                            _evaluate_metrics,
                            experiment_name,
                            dataset_name,
                            model_name,
                            valuation_method_name,
                            repetition_id,
                            metric_name,
                        )

            if stages["render_plots"]:
                logger.info(f"Render plots for {experiment_name} and {model_name}.")
                _render_plots(experiment_name, model_name)

    except KeyboardInterrupt:
        logger.info("Interrupted by Ctrl+C.")
    else:
        logger.info("Shutdown system.")
        os.system("sudo shutdown now")


if __name__ == "__main__":
    run_pipeline()
