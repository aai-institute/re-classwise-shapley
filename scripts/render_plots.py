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
    plot_value_decay,
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
    repetitions = list(range(repetitions["from"], repetitions["to"] + 1))
    curves_def = params["experiments"][experiment_name]["curves"]
    curve_names = list(curves_def.keys())
    metrics_def = params["experiments"][experiment_name]["metrics"]
    metric_names = list(metrics_def.keys())

    mlflow.set_tracking_uri(params["settings"]["mlflow_tracking_uri"])
    experiment_id = get_or_create_mlflow_experiment(mlflow_id)
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Starting experiment with id `{experiment_id}.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        logger.info("Flatten parameters & upload to mlflow.")
        mlflow.log_params(flatten_dict(params))

        logger.info("Record datasets in mlflow.")
        log_datasets(
            Accessor.datasets(
                experiment_name,
                dataset_names,
            )
        )

        plt.switch_backend("agg")

        params = load_params_fast()
        plot_format = params["settings"]["plot_format"]

        logger.info(f"Load valuations results.")
        valuation_results = Accessor.valuation_results(
            experiment_name,
            model_name,
            dataset_names,
            repetitions,
            method_names,
        )
        logger.info(f"Plotting value decay for all methods.")
        with plot_value_decay(valuation_results, method_names) as fig:
            log_figure(fig, output_folder, f"decay.{plot_format}", "values")

        for method_name in method_names:
            logger.info(f"Plot histogram for values of method `{method_name}`.")
            with plot_histogram(valuation_results, [method_name]) as fig:
                log_figure(
                    fig, output_folder, f"density.{method_name}.{plot_format}", "values"
                )

        threshold_characteristics_settings = params["settings"][
            "threshold_characteristics"
        ]
        if threshold_characteristics_settings.get("active", False):
            logger.info(f"Load threshold characteristics.")
            plot_threshold_characteristics_results = (
                Accessor.threshold_characteristics_results(
                    experiment_name,
                    dataset_names,
                    repetitions,
                )
            )

            logger.info(f"Plot threshold characteristics.")
            with plot_threshold_characteristics(
                plot_threshold_characteristics_results
            ) as fig:
                log_figure(
                    fig,
                    output_folder,
                    f"threshold_characteristics.{plot_format}",
                    "threshold_characteristics",
                )

        params = load_params_fast()
        time_settings = params["settings"]["time"]
        if time_settings.get("active", False):
            logger.info("Plot boxplot for execution time.")
            with plot_time(valuation_results) as fig:
                log_figure(fig, output_folder, f"time.{plot_format}", "boxplots")

        logger.info("Loading curves form hard disk.")
        loaded_curves = Accessor.curves(
            experiment_name,
            model_name,
            dataset_names,
            method_names,
            curve_names,
            repetitions,
        )
        for curve_name in curve_names:
            logger.info(f"Processing curve '{curve_name}'.")
            selected_loaded_curves = loaded_curves.loc[
                loaded_curves["curve_name"] == curve_name
            ].copy()
            curve_def = curves_def[curve_name]
            for plot_settings_name in curve_def["plots"]:
                plot_settings = params["plots"][plot_settings_name]
                logger.info(
                    f"Plotting {plot_settings['type']} plot with name '{plot_settings_name}'"
                )
                match plot_settings["type"]:
                    case "line":
                        plot_perc = plot_settings.get("plot_perc", 1.0)
                        x_label = plot_settings.get("x_label", None)
                        y_label = plot_settings.get("y_label", None)
                        agg = plot_settings.get("agg", "mean")
                        with plot_curves(
                            selected_loaded_curves,
                            plot_perc=plot_perc,
                            x_label=x_label,
                            y_label=y_label,
                        ) as fig:
                            log_figure(
                                fig,
                                output_folder,
                                f"{curve_name}.{plot_format}",
                                "curves",
                            )
                    case _:
                        raise NotImplementedError

            logger.info("Loading metrics form hard disk.")
            loaded_metrics = Accessor.metrics(
                experiment_name,
                model_name,
                dataset_names,
                method_names,
                metric_names,
                repetitions,
                curve_name,
            )
            for metric_name in metric_names:
                logger.info(f"Processing metric '{metric_name}'.")
                selected_loaded_metrics = loaded_metrics.loc[
                    loaded_metrics["metric_name"] == metric_name
                ].copy()
                metric_def = metrics_def[metric_name]
                for plot_settings_name in metric_def["plots"]:
                    plot_settings = params["plots"][plot_settings_name]
                    logger.info(
                        f"Plotting {plot_settings['type']} plot with name '{plot_settings_name}'"
                    )
                    match plot_settings["type"]:
                        case "table":
                            logger.info(
                                f"Converting df to table for metric '{metric_name}'."
                            )
                            metric_table = linear_dataframe_to_table(
                                selected_loaded_metrics,
                                "dataset_name",
                                "method_name",
                                "metric",
                                np.mean,
                            )
                            for dataset_name, row in metric_table.items():
                                for method_name, v in row.items():
                                    mlflow.log_metric(
                                        f"{metric_name}.{dataset_name}.{method_name}", v
                                    )

                            logger.info(f"Plotting table for metric '{metric_name}'.")
                            with plot_metric_table(metric_table) as fig:
                                log_figure(
                                    fig,
                                    output_folder,
                                    f"{metric_name}.{curve_name}.table.{plot_format}",
                                    "tables",
                                )
                        case "boxplot":
                            x_label = plot_settings.get("x_label", None)
                            logger.info(f"Plotting boxplot for metric '{metric_name}'.")
                            with plot_metric_boxplot(
                                selected_loaded_metrics, x_label=x_label
                            ) as fig:
                                log_figure(
                                    fig,
                                    output_folder,
                                    f"{metric_name}.{curve_name}.box.{plot_format}",
                                    "boxplots",
                                )

                        case _:
                            raise NotImplementedError

    logger.info(f"Finished rendering plots and metrics.")


if __name__ == "__main__":
    load_dotenv()
    render_plots()
