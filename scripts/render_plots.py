"""
Stage six renders the plots and stores them on disk and in mlflow.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Render plots for the data valuation experiment. The plots are stored in the
`Accessor.PLOT_PATH` directory. The plots are stored in `*.svg` format. The plots are
also stored in mlflow. The id of the mlflow experiment is given by the schema
`experiment_name.model_name`.
"""
import os
import os.path
from datetime import datetime

import click
import mlflow
from dotenv import load_dotenv

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import log_datasets, log_figure, setup_logger
from re_classwise_shapley.plotting import plot_curves, plot_histogram, plot_time
from re_classwise_shapley.utils import flatten_dict, load_params_fast

logger = setup_logger("render_plots")


def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Get or create a mlflow experiment. If the experiment does not exist, it will be
    created.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Identifier of the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def render_plots(experiment_name: str, model_name: str):
    """
    Render plots for the data valuation experiment. The plots are stored in the
    `Accessor.PLOT_PATH` directory. The plots are stored in `*.svg` format.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        model_name: Model to use. As specified in the `params.models` section.
    """
    load_dotenv()
    logger.info("Starting plotting of data valuation experiment")
    output_folder = Accessor.PLOT_PATH / experiment_name / model_name
    mlflow_id = f"{experiment_name}.{model_name}"

    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    method_names = params_active["valuation_methods"]
    repetitions = params_active["repetitions"]
    metrics = list(params["experiments"][experiment_name]["metrics"].keys())

    mlflow.set_tracking_uri(params["settings"]["mlflow_tracking_uri"])
    experiment_id = get_or_create_mlflow_experiment(mlflow_id)
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Starting run.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        logger.info("Log parameters.")
        mlflow.log_params(flatten_dict(params))
        logger.info("Log datasets.")
        log_datasets(
            Accessor.datasets(
                experiment_name,
                dataset_names,
                repetitions,
            )
        )

        valuation_results = Accessor.valuation_results(
            experiment_name,
            model_name,
            dataset_names,
            repetitions,
            method_names,
        )
        for method_name in method_names:
            logger.info(f"Plot histogram for method {method_name} values.")
            with plot_histogram(valuation_results, [method_name, "tmc_shapley"]) as fig:
                log_figure(
                    fig, output_folder, f"density.{method_name}.svg", "densities"
                )

        logger.info(f"Plot boxplot for execution time.")
        with plot_time(valuation_results) as fig:
            log_figure(fig, output_folder, "time.svg")

        metrics_and_curves = Accessor.metrics_and_curves(
            experiment_name,
            model_name,
            dataset_names,
            method_names,
            repetitions,
            metrics,
        )
        for metric_name in metrics:
            logger.info(f"Plotting curve for metric {metric_name}.")
            single_curve = metrics_and_curves.loc[
                metrics_and_curves["metric_name"] == metric_name
            ]
            with plot_curves(single_curve) as fig:
                log_figure(fig, output_folder, f"{metric_name}.svg", "curves")


if __name__ == "__main__":
    render_plots()
