import os
import os.path
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from numpy._typing import NDArray
from PIL import Image
from pydvl.value import ValuationResult
from scipy.stats import gaussian_kde

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import shaded_mean_normal_confidence_interval
from re_classwise_shapley.utils import flatten_dict, load_params_fast

logger = setup_logger("render_plots")

# Mapping from method names to single colors
COLOR_ENCODING = {
    "random": "black",
    "beta_shapley": "blue",
    "loo": "orange",
    "tmc_shapley": "green",
    "classwise_shapley": "red",
    "owen_sampling_shapley": "purple",
    "banzhaf_shapley": "turquoise",
    "least_core": "gray",
}

# Mapping from colors to mean and shade color.
COLORS = {
    "black": ("black", "silver"),
    "blue": ("dodgerblue", "lightskyblue"),
    "orange": ("darkorange", "gold"),
    "green": ("limegreen", "seagreen"),
    "red": ("indianred", "firebrick"),
    "purple": ("darkorchid", "plum"),
    "gray": ("gray", "lightgray"),
    "turquoise": ("turquoise", "lightcyan"),
}


def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Get or create a mlflow experiment. If the experiment does not exist, it will be
    created.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Id of the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(
            experiment_name,
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def render_plots(experiment_name: str, model_name: str):
    load_dotenv()
    logger.info("Starting plotting of data valuation experiment")
    experiment_path = Accessor.RESULT_PATH / experiment_name / model_name
    output_folder = Accessor.PLOT_PATH / experiment_name / model_name
    mlflow_id = f"{experiment_name}.{model_name}"
    mlflow.set_tracking_uri("http://localhost:5000")

    experiment_id = get_or_create_mlflow_experiment(mlflow_id)
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Starting run.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        logger.info("Log params.")
        params = load_params_fast()
        params = flatten_dict(params)
        mlflow.log_params(params)

        logger.info("Plotting histogram.")
        plot_histogram(experiment_name, model_name, output_folder)

        params = load_params_fast()
        metrics = params["experiments"][experiment_name]["metrics"]
        logger.info("Plotting metric tables.")
        results_per_dataset = load_results_per_dataset_and_method(
            experiment_path, metrics
        )
        plot_metric_table(results_per_dataset, output_folder)

        logger.info("Plotting curves.")
        plot_curves(
            results_per_dataset,
            f"Experiment '{experiment_name}' on model '{model_name}'",
            output_folder,
        )


def mean_density(
    values: NDArray[ValuationResult],
    hist_range: Tuple[float, float],
    bins: int = 50,
) -> NDArray[np.float_]:
    """
    Compute the mean and confidence interval of a set of `ValuationResult` objects.
    Args:
        values: List of `ValuationResult` objects.
        hist_range: Tuple of minimum hist rand and maximum hist range.
        bins: Number of bins to be used for histogram.

    Returns:
        A tuple with the mean and 95% confidence interval.
    """
    x = np.linspace(*hist_range, bins)
    kde = gaussian_kde(values.reshape(-1))
    density = kde(x)
    return density


def plot_histogram(experiment_name: str, model_name: str, output_folder: Path):
    """
    Plot the histogram of the data values for each dataset and valuation method.

    Args:
        experiment_name: Experiment name to obtain histograms of.
        model_name: Model name to obtain histograms of.
        output_folder: Folder to store the plots in.

    """
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    repetitions = params_active["repetitions"]
    bins = 20

    accessor = Accessor(experiment_name, model_name)
    valuation_results = accessor.valuation_results(
        dataset_names, valuation_methods, repetitions
    )

    figsize = (4, 3)
    h = 2
    w = int((len(valuation_results) + 1) / 2)
    fig_ax_d = {}
    idx = 0
    for dataset_name, dataset_config in valuation_results.items():
        for method_name, method_values in dataset_config.items():
            logger.info(f"Plotting histogram for {dataset_name=}, {method_name=}.")

            if method_name not in fig_ax_d.keys():
                fig, ax = plt.subplots(h, w, figsize=(w * figsize[0], h * figsize[1]))
                ax = ax.T.flatten()
                fig_ax_d[method_name] = (fig, ax)

            fig, ax = fig_ax_d[method_name]
            sns.histplot(method_values.reshape(-1), kde=True, ax=ax[idx])
            ax[idx].set_title(f"({chr(97 + idx)}) {dataset_name}")

        idx += 1

    num_datasets = len(valuation_results)
    for key, (fig, ax) in fig_ax_d.items():
        if len(ax) == num_datasets + 1:
            ax[-1].grid(False)
            ax[-1].axis("off")

        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f"Density plot values by '{key}'")
        f_name = f"density.{key}.png"
        logger.info(f"Logging plot '{f_name}'")
        output_file = output_folder / f_name
        mlflow.log_figure(fig, f"densities/{f_name}")
        fig.savefig(output_file)


def plot_curves(
    results_per_dataset: Dict[
        str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]
    ],
    title: str,
    output_folder: Path,
):
    """
    Plot the curves of the data values for each dataset and valuation method.

    Args:
        results_per_dataset: A dictionary of dictionaries containing realizations of
            the distribution over curves.
        title: Title of the plot.
        output_folder: Output folder to store the plots in.
    """
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    metric_names = sorted(
        results_per_dataset[dataset_names[0]][valuation_methods[0]].keys()
    )
    output_folder = output_folder / "curves"
    os.makedirs(output_folder, exist_ok=True)

    num_datasets = len(dataset_names)
    h = 2
    w = int(num_datasets / 2) + 1
    for metric_name in metric_names:
        fig, ax = plt.subplots(h, w, figsize=(w * 20 / 4, h * 5 / 2))
        ax = ax.T.flatten()

        for idx, dataset_name in enumerate(dataset_names):
            for method_name in valuation_methods:
                logger.info(
                    f"Plotting curve for {dataset_name=}, {metric_name=}, {method_name=}."
                )
                results = results_per_dataset[dataset_name][method_name][metric_name][1]
                color_name = COLOR_ENCODING[method_name]
                mean_color, shade_color = COLORS[color_name]
                results = results.sort_index()
                shaded_mean_normal_confidence_interval(
                    results,
                    abscissa=results.index,
                    mean_color=mean_color,
                    shade_color=shade_color,
                    label=method_name,
                    ax=ax[idx],
                )
                ax[idx].set_title(f"({chr(97 + idx)}) {dataset_name}")

        handles, labels = ax[num_datasets - 1].get_legend_handles_labels()
        if len(ax) == num_datasets + 2:
            ax[-2].grid(False)
            ax[-2].axis("off")

        ax[-1].grid(False)
        ax[-1].axis("off")
        ax[-1].legend(handles, labels, loc="center", fontsize=16)
        fig.suptitle(title + f" with metric '{metric_name}'")
        fig.subplots_adjust(hspace=0.4)

        output_file = output_folder / f"{metric_name}.png"
        mlflow.log_figure(fig, f"curves/{metric_name}.png")
        fig.savefig(output_file)


def plot_metric_table(
    results_per_dataset: Dict[
        str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]
    ],
    output_folder: Path,
):
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    metric_names = sorted(
        results_per_dataset[dataset_names[0]][valuation_methods[0]].keys()
    )
    output_folder = output_folder / "metrics"
    os.makedirs(output_folder, exist_ok=True)

    for metric_name in metric_names:
        mean_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)
        std_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)

        for dataset_name in dataset_names:
            for method_name in valuation_methods:
                logger.info(
                    f"Plotting metric for {dataset_name=}, {metric_name=}, {method_name=}."
                )
                val = results_per_dataset[dataset_name][method_name][metric_name][
                    0
                ].loc[metric_name]
                m = np.mean(val)
                s = np.std(val)
                mean_metric.loc[dataset_name, method_name] = m
                std_metric.loc[dataset_name, method_name] = s
                mlflow.log_metric(f"{metric_name}_{dataset_name}_{method_name}_mean", m)
                mlflow.log_metric(f"{metric_name}_{dataset_name}_{method_name}_std", s)

        df_styled = mean_metric.style.highlight_max(color="lightgreen", axis=1)
        output_file = output_folder / f"{metric_name}_mean.png"
        dfi.export(df_styled, output_file, table_conversion="matplotlib")
        with Image.open(output_file) as im:
            mlflow.log_image(im, f"metrics/{metric_name}_mean.png")

        df_styled = std_metric.style.highlight_min(color="lightgreen", axis=1)
        output_file = output_folder / f"{metric_name}_std.png"
        dfi.export(df_styled, output_file, table_conversion="matplotlib")
        with Image.open(output_file) as im:
            mlflow.log_image(im, f"metrics/{metric_name}_std.png")


def load_results_per_dataset_and_method(
    experiment_path: Path, metrics: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]]:
    """
    Load the results per dataset and method. The results are loaded from the
    `experiment_path` directory. The results are loaded from the `metric_names` files.

    Args:
        experiment_path: Path to the experiment directory.
        metrics: List of metric names to load.

    Returns:

    """
    params = load_params_fast()
    params_active = params["active"]
    repetitions = params_active["repetitions"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    curves_per_dataset = {}

    for dataset_name in dataset_names:
        dataset_path = experiment_path / dataset_name

        curves_per_valuation_method = {}

        for valuation_method in valuation_methods:
            curves_per_metric = {}
            for repetition in repetitions:
                repetition_path = dataset_path / f"{repetition}" / valuation_method
                for key, metric_config in metrics.items():
                    if key not in curves_per_metric:
                        curves_per_metric[key] = []

                    logger.info(f"Loading metric {key} from path '{repetition_path}'.")
                    metric = pd.read_csv(repetition_path / f"{key}.csv")
                    metric = metric.drop(columns=[metric.columns[0]])
                    metric.index = [key]
                    metric.index.name = key
                    metric.columns = [repetition]

                    curve = pd.read_csv(repetition_path / f"{key}.curve.csv")
                    curve.index = curve[curve.columns[0]]
                    curve = curve.drop(columns=[curve.columns[0]])

                    len_curve_perc = metric_config["len_curve_perc"]
                    curve = curve.iloc[: int(len_curve_perc * len(curve))]
                    curves_per_metric[key].append((metric, curve))

            curves_per_metric = {
                k: (
                    pd.concat([t[0] for t in v], axis=1),
                    pd.concat([t[1] for t in v], axis=1),
                )
                for k, v in curves_per_metric.items()
            }
            curves_per_valuation_method[valuation_method] = curves_per_metric

        curves_per_dataset[dataset_name] = curves_per_valuation_method

    return curves_per_dataset


if __name__ == "__main__":
    render_plots()
