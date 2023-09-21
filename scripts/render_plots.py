import os
import os.path
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from dvc.api import params_show
from numpy._typing import NDArray
from PIL import Image
from pydvl.value import ValuationResult
from scipy.stats import gaussian_kde

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import shaded_mean_normal_confidence_interval

logger = setup_logger()

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


def get_or_create_mlflow_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(
            experiment_name,
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def flatten_dict(d, parent_key="", separator="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, separator=separator))
        else:
            items[new_key] = v
    return items


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


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
        params = params_show()
        params = flatten_dict(params)
        mlflow.log_params(params)

        logger.info("Plotting histogram.")
        plot_histogram(experiment_name, model_name, output_folder)

        # plot curves and metrics
        params = params_show()
        metric_names = [
            k
            for k, v in params["experiments"][experiment_name]["metrics"].items()
            if v["type"] == "metric"
        ]
        logger.info("Plotting metric tables.")
        metrics_per_dataset = load_metrics_per_dataset(experiment_path, metric_names)
        plot_metric_table(metrics_per_dataset, output_folder)

        curve_names = [
            k
            for k, v in params["experiments"][experiment_name]["metrics"].items()
            if v["type"] == "curve"
        ]
        curves_per_dataset = load_metrics_per_dataset(experiment_path, curve_names)
        logger.info("Plotting curves.")
        plot_curves(
            curves_per_dataset,
            f"Experiment '{experiment_name}' on model '{model_name}'",
            output_folder,
        )


def mean_and_confidence_interval(
    values: NDArray[ValuationResult],
    hist_range: Tuple[int, int],
    bins: int = 50,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    x = np.linspace(*hist_range, bins)
    densities = []
    for v in values:
        kde = gaussian_kde(v, bw_method=0.05)
        density = kde(x)
        densities.append(density)

    mean = np.mean(densities, axis=0)
    std = 1.96 * np.std(densities, axis=0) / np.sqrt(len(densities))
    return mean, std


def plot_histogram(experiment_name: str, model_name: str, output_folder: Path):
    params = params_show()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    repetitions = params_active["repetitions"]
    bins = 20

    accessor = Accessor(experiment_name, model_name)
    valuation_results = accessor.valuation_results(
        dataset_names, valuation_methods, repetitions
    )
    min_value = min(np.min(v) for d in valuation_results.values() for v in d.values())
    max_value = max(np.max(v) for d in valuation_results.values() for v in d.values())
    hist_range = (min_value, max_value)

    figsize = (4, 3)
    h = 2
    w = int(len(valuation_results) / 2)
    fig_ax_d = {}
    idx = 0
    for dataset_name, dataset_config in valuation_results.items():
        for method_name, method_values in dataset_config.items():
            logger.info(f"Plotting histogram for {dataset_name=}, {method_name=}.")

            if method_name not in fig_ax_d.keys():
                fig, ax = plt.subplots(h, w, figsize=(w * figsize[0], h * figsize[1]))
                ax = ax.T.flatten()
                fig_ax_d[method_name] = (fig, ax)

            mean_bars, std_bars = mean_and_confidence_interval(
                method_values, hist_range=hist_range, bins=bins
            )

            # Plot the mean histogram
            bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_color, std_color = COLORS[COLOR_ENCODING[method_name]]
            fig, ax = fig_ax_d[method_name]
            ax[idx].plot(
                bin_centers,
                mean_bars,
                color=mean_color,
                alpha=0.8,
            )
            ax[idx].fill_between(
                bin_centers,
                mean_bars - std_bars,
                mean_bars + std_bars,
                color=std_color,
                alpha=0.3,
            )
            ax[idx].set_title(f"({chr(97 + idx)}) {dataset_name}")

        idx += 1

    num_datasets = len(valuation_results)
    for key, (fig, ax) in fig_ax_d.items():
        if len(ax) == num_datasets + 1:
            ax[-1].grid(False)
            ax[-1].axis("off")

        fig.subplots_adjust(hspace=0.4)
        output_file = output_folder / f"density.{key}.png"
        mlflow.log_figure(fig, f"density.{key}.png")
        fig.savefig(output_file)


def plot_curves(results_per_dataset: Dict, title: str, output_folder: Path):
    """
    Plot the metric curves for each dataset and valuation method.
    :param results_per_dataset:
    :param title:
    :return:
    """
    params = params_show()
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
        if metric_name == "time":
            continue

        fig, ax = plt.subplots(h, w, figsize=(w * 20 / 4, h * 5 / 2))
        ax = ax.T.flatten()

        for idx, dataset_name in enumerate(dataset_names):
            for method_name in valuation_methods:
                logger.info(
                    f"Plotting curve for {dataset_name=}, {metric_name=}, {method_name=}."
                )
                results = results_per_dataset[dataset_name][method_name][metric_name]
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


def plot_metric_table(results_per_dataset, output_folder: Path):
    params = params_show()
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
                val = results_per_dataset[dataset_name][method_name][metric_name].loc[
                    metric_name
                ]
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

        df_styled = std_metric.style.highlight_max(color="lightgreen", axis=1)
        output_file = output_folder / f"{metric_name}_std.png"
        dfi.export(df_styled, output_file, table_conversion="matplotlib")
        with Image.open(output_file) as im:
            mlflow.log_image(im, f"metrics/{metric_name}_std.png")


def load_metrics_per_dataset(experiment_path: Path, metric_names: List[str] = None):
    params = params_show()
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
                for key in metric_names:
                    if key not in curves_per_metric:
                        curves_per_metric[key] = []

                    new_element = pd.read_csv(repetition_path / f"{key}.csv")
                    first_col = new_element.columns[0]
                    new_element.index = new_element[first_col]
                    new_element = new_element.drop(columns=[first_col])

                    if len(new_element) == 1:
                        new_element.index = [key]
                        new_element.index.name = key
                        new_element.columns = [repetition]

                    curves_per_metric[key].append(new_element)

            curves_per_metric = {
                k: pd.concat(v, axis=1) for k, v in curves_per_metric.items()
            }
            curves_per_valuation_method[valuation_method] = curves_per_metric

        curves_per_dataset[dataset_name] = curves_per_valuation_method

    return curves_per_dataset


if __name__ == "__main__":
    render_plots()
