import os
from pathlib import Path
from typing import Dict, Optional

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dvc.api import params_show

from re_classwise_shapley.config import Config
from re_classwise_shapley.experiments import ExperimentResult
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import plot_curve, setup_plotting

logger = setup_logger()

COLOR_ENCODING = {
    "random": "black",
    "beta_shapley": "blue",
    "loo": "orange",
    "tmc_shapley": "green",
    "classwise_shapley": "red",
    "classwise_shapley_add_idx": "purple",
}


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--sub-folder", type=str, required=False)
def render_plots(
    experiment_name: str, model_name: str, sub_folder: Optional[str] = None
):
    logger.info("Starting plotting of data valuation experiment")
    setup_plotting()

    experiment_path = Config.RESULT_PATH / experiment_name / model_name
    output_dir = Config.PLOT_PATH / experiment_name / model_name
    os.makedirs(output_dir, exist_ok=True)

    if sub_folder is not None:
        experiment_path /= sub_folder
        output_dir /= sub_folder

    results_per_dataset = load_results_per_dataset(experiment_path)
    plot_metric_table(results_per_dataset, output_dir)
    plot_metric_curves(
        results_per_dataset,
        f"Experiment '{experiment_name}' on model '{model_name}'",
        output_dir,
    )


def plot_metric_curves(results_per_dataset: Dict, title: str, output_dir: Path):
    """
    Plot the metric curves for each dataset and valuation method.
    :param results_per_dataset:
    :param title:
    :param output_dir:
    :return:
    """
    params = params_show()
    dataset_names = params["active"]["datasets"]
    num_datasets = len(dataset_names)
    valuation_method_names = results_per_dataset[dataset_names[0]][
        0
    ].valuation_method_names
    metric_names = results_per_dataset[dataset_names[0]][0].metric_names
    h = 2
    w = int(num_datasets / 2) + 1
    for metric_name in metric_names:
        fig, ax = plt.subplots(h, w, figsize=(20, 6))
        ax = np.array(ax).flatten()

        for idx, dataset_name in enumerate(dataset_names):
            cax = ax[idx]
            dataset_results = results_per_dataset[dataset_name]
            d = {}
            for valuation_method_name in valuation_method_names:
                d[valuation_method_name] = (
                    pd.concat(
                        [
                            result.curves[valuation_method_name][metric_name]
                            for result in dataset_results
                        ],
                        axis=1,
                    ).sort_index(),
                    {
                        "color": COLOR_ENCODING[valuation_method_name],
                        "plot_single": metric_name == "density",
                    },
                )

            plot_curve(d, title=dataset_name, ax=cax)
            handles, labels = cax.get_legend_handles_labels()

        ax[-1].grid(False)
        ax[-1].axis("off")
        ax[-1].legend(handles, labels, loc="center", fontsize=16)
        fig.suptitle(title + f" with metric '{metric_name}'")
        fig.subplots_adjust(hspace=0.3)
        fig.savefig(output_dir / f"curve_{metric_name}.png")


def plot_metric_table(results_per_dataset, output_dir):
    params = params_show()
    dataset_names = params["active"]["datasets"]
    valuation_method_names = results_per_dataset[dataset_names[0]][
        0
    ].valuation_method_names
    metric_names = results_per_dataset[dataset_names[0]][0].metric_names
    for metric_name in metric_names:
        mean_metric = pd.DataFrame(index=dataset_names, columns=valuation_method_names)

        for dataset_name in dataset_names:
            for valuation_method_name in valuation_method_names:
                m = np.mean(
                    [
                        result.metric[valuation_method_name][metric_name]
                        for result in results_per_dataset[dataset_name]
                    ]
                )
                mean_metric.loc[dataset_name, valuation_method_name] = m

        df_styled = mean_metric.style.highlight_max(color="lightgreen", axis=1)
        dfi.export(df_styled, output_dir / f"metrics_{metric_name}.png")


def load_results_per_dataset(experiment_path: Path):
    params = params_show()
    dataset_names = params["active"]["datasets"]
    results_per_dataset = {}
    for dataset_name in dataset_names:
        dataset_path = experiment_path / dataset_name
        n_repetitions = len(
            [f for f in os.listdir(dataset_path) if f.startswith("repetition")]
        )

        results = []
        for repetition in range(n_repetitions):
            experiment_result = ExperimentResult.load(
                experiment_path / dataset_name / f"{repetition=}"
            )
            results.append(experiment_result)

        results_per_dataset[dataset_name] = results
    return results_per_dataset


if __name__ == "__main__":
    render_plots()
