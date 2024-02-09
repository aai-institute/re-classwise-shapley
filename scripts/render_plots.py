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
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import (
    get_or_create_mlflow_experiment,
    log_datasets,
    log_figure,
    setup_logger,
)
from re_classwise_shapley.plotting import (
    plot_curves,
    plot_histogram,
    plot_metric_boxplot,
    plot_metric_table,
    plot_threshold_characteristics,
    plot_time,
)
from re_classwise_shapley.utils import (
    flatten_dict,
    linear_dataframe_to_table,
    load_params_fast,
)

logger = setup_logger("render_plots")


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
    _render_plots(experiment_name, model_name)


def _render_plots(experiment_name: str, model_name: str):
    logger.info("Starting plotting of data valuation experiment")
    output_folder = Accessor.PLOT_PATH / experiment_name / model_name
    mlflow_id = f"{experiment_name}.{model_name}"

    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    method_names = params_active["valuation_methods"]
    repetitions = params_active["repetitions"]
    metrics_def = params["experiments"][experiment_name]["metrics"]
    metrics = list(metrics_def.keys())

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
            )
        )

        plt.switch_backend("agg")
        logger.info(f"Plot threshold characteristics.")
        plot_threshold_characteristics_results = (
            Accessor.threshold_characteristics_results(
                experiment_name,
                dataset_names,
                repetitions,
            )
        )

        with plot_threshold_characteristics(
            plot_threshold_characteristics_results
        ) as fig:
            log_figure(
                fig,
                output_folder,
                f"threshold_characteristics.svg",
                "threshold_characteristics",
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
            with plot_histogram(valuation_results, [method_name]) as fig:
                log_figure(
                    fig, output_folder, f"density.{method_name}.svg", "densities"
                )

        logger.info("Plot boxplot for execution time.")
        with plot_time(valuation_results) as fig:
            log_figure(fig, output_folder, "time.svg", "boxplots")

        metrics_and_curves = Accessor.metrics_and_curves(
            experiment_name,
            model_name,
            dataset_names,
            method_names,
            repetitions,
            metrics,
        )
        for metric_name in metrics:
            metric_and_curves_for_metric = metrics_and_curves.loc[
                metrics_and_curves["metric_name"] == metric_name
            ].copy()

            len_curve_perc = metrics_def[metric_name].pop("len_curve_perc", None)
            curve_label = metrics_def[metric_name].pop("curve_label", None)
            y_label = metrics_def[metric_name].pop("y_label", None)
            logger.info(f"Plotting curve for metric {metric_name}.")
            with plot_curves(
                metric_and_curves_for_metric,
                len_curve_perc=len_curve_perc,
                x_label=curve_label,
                y_label=y_label,
            ) as fig:
                log_figure(fig, output_folder, f"{metric_name}.svg", "curves")

            logger.info(f"Plotting table for metric {metric_name}.")
            metric_table = linear_dataframe_to_table(
                metric_and_curves_for_metric,
                "dataset_name",
                "method_name",
                "metric",
                np.mean,
            )
            for dataset_name, row in metric_table.items():
                for method_name, v in row.items():
                    mlflow.log_metric(f"{metric_name}.{dataset_name}.{method_name}", v)

            with plot_metric_table(metric_table) as fig:
                log_figure(fig, output_folder, f"{metric_name}.table.svg", "tables")

            metric_label = metrics_def[metric_name].pop("metric_label", None)
            logger.info(f"Plotting boxplot for metric {metric_name}.")
            with plot_metric_boxplot(
                metric_and_curves_for_metric, x_label=metric_label
            ) as fig:
                log_figure(fig, output_folder, f"{metric_name}.box.svg", "boxplots")


if __name__ == "__main__":
    load_dotenv()
    render_plots()
