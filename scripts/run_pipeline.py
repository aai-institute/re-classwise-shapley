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
from fetch_data import _fetch_data
from preprocess_data import _preprocess_data
from render_plots import _render_plots
from sample_data import _sample_data

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("preprocess_data")


@click.command()
def run_pipeline():
    """
    Runs the whole pipeline without dvc.
    """
    params = load_params_fast()
    active_params = params["active"]

    for dataset_name in active_params["datasets"]:
        _fetch_data(dataset_name)
        _preprocess_data(dataset_name)

    for (
        experiment_name,
        dataset_name,
        repetition_id,
    ) in product(
        active_params[k]
        for k in [
            "experiments",
            "datasets",
            "repetitions",
        ]
    ):
        _sample_data(experiment_name, dataset_name, repetition_id)

    for experiment_name, model_name in product(
        active_params[k]
        for k in [
            "experiments",
            "models",
        ]
    ):
        for (
            dataset_name,
            valuation_method_name,
            repetition_id,
        ) in product(
            active_params[k]
            for k in [
                "datasets",
                "valuation_methods",
                "repetitions",
            ]
        ):
            _calculate_values(
                experiment_name,
                dataset_name,
                model_name,
                valuation_method_name,
                repetition_id,
            )

        _render_plots(experiment_name, model_name)


if __name__ == "__main__":
    run_pipeline()
