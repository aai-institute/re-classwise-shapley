import os
import os.path
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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
    experiment_path = Config.RESULT_PATH / experiment_name / model_name
    output_folder = Config.PLOT_PATH / experiment_name / model_name
    mlflow_id = f"{experiment_name}.{model_name}"
    mlflow.set_tracking_uri("http://localhost:5000")

    experiment_id = get_or_create_mlflow_experiment(mlflow_id)
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Starting run.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        plot_histogram(experiment_name, model_name, output_folder)

        logger.info("Log params.")
        params = params_show()
        params = flatten_dict(params)
        mlflow.log_params(params)

        logger.info("Packing results.")
        filename = f"results"
        shutil.make_archive(filename, "tar", experiment_path)
        logger.info("Uploading results.")
        tar_filename = f"{filename}.tar"
        mlflow.log_artifact(tar_filename)
        os.remove(tar_filename)

        # plot curves and metrics
        results_per_dataset = load_results_per_dataset(experiment_path)
        metrics_per_dataset = {
            dataset: {method: v["metrics"] for method, v in method_config.items()}
            for dataset, method_config in results_per_dataset.items()
        }
        plot_metric_table(metrics_per_dataset, output_folder)
        results_per_dataset = {
            dataset: {
                method: {
                    metric: v
                    for metric, v in metric_config.items()
                    if metric != "metrics"
                }
                for method, metric_config in method_config.items()
            }
            for dataset, method_config in results_per_dataset.items()
        }
        plot_metric_curves(
            results_per_dataset,
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
        kde = gaussian_kde(v)
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
    w = int(len(valuation_results) / 2) + 1
    fig, ax = plt.subplots(h, w, figsize=(w * figsize[0], h * figsize[1]))
    ax = ax.T.flatten()
    idx = 0
    for dataset_name, dataset_config in valuation_results.items():
        for method_name, method_values in dataset_config.items():
            mean_bars, std_bars = mean_and_confidence_interval(
                method_values, hist_range=hist_range, bins=bins
            )

            # Plot the mean histogram
            bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_color, std_color = COLORS[COLOR_ENCODING[method_name]]
            sns.histplot(
                data=method_values.reshape(-1),
                multiple="layer",
                kde=True,
                ax=ax[idx],
                color=mean_color,
                label="Mean Histogram",
                alpha=0.7,
            )
            ax[idx].fill_between(
                bin_centers,
                mean_bars - std_bars,
                mean_bars + std_bars,
                color=std_color,
                alpha=0.3,
            )

        idx += 1

    output_file = output_folder / f"density.png"
    mlflow.log_figure(fig, "density.png")
    fig.savefig(output_file)


def plot_metric_curves(results_per_dataset: Dict, title: str, output_folder: Path):
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

        logger.info(f"Plotting metric {metric_name}.")
        fig, ax = plt.subplots(h, w, figsize=(w * 20 / 4, h * 5 / 2))
        ax = ax.T.flatten()

        for idx, dataset_name in enumerate(dataset_names):
            logger.info(f"Plotting dataset {dataset_name}.")

            for method_name in valuation_methods:
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
        fig.subplots_adjust(hspace=0.3)

        output_file = output_folder / f"{metric_name}.png"
        mlflow.log_figure(fig, f"curves/{metric_name}.png")
        fig.savefig(output_file)


def plot_metric_table(results_per_dataset, output_folder: Path):
    params = params_show()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    metric_names = sorted(
        results_per_dataset[dataset_names[0]][valuation_methods[0]].index
    )
    output_folder = output_folder / "metrics"
    os.makedirs(output_folder, exist_ok=True)

    for metric_name in metric_names:
        mean_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)
        std_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)

        for dataset_name in dataset_names:
            for valuation_method_name in valuation_methods:
                val = results_per_dataset[dataset_name][valuation_method_name].loc[
                    metric_name
                ]
                m = np.mean(val)
                s = np.std(val)
                mean_metric.loc[dataset_name, valuation_method_name] = m
                std_metric.loc[dataset_name, valuation_method_name] = s
                mlflow.log_metric(
                    f"{metric_name}_{dataset_name}_{valuation_method_name}_mean", m
                )
                mlflow.log_metric(
                    f"{metric_name}_{dataset_name}_{valuation_method_name}_std", s
                )

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


def load_results_per_dataset(experiment_path: Path):
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
                repetition_files = os.listdir(repetition_path)
                for repetition_file in repetition_files:
                    key = ".".join(repetition_file.split(".")[:-1])
                    if key not in curves_per_metric:
                        curves_per_metric[key] = []

                    new_element = pd.read_csv(repetition_path / repetition_file)
                    first_col = new_element.columns[0]
                    new_element.index = new_element[first_col]
                    new_element = new_element.drop(columns=[first_col])
                    curves_per_metric[key].append(new_element)

            curves_per_metric = {
                k: pd.concat(v, axis=1) for k, v in curves_per_metric.items()
            }
            curves_per_valuation_method[valuation_method] = curves_per_metric

        curves_per_dataset[dataset_name] = curves_per_valuation_method

    return curves_per_dataset


if __name__ == "__main__":
    render_plots()
