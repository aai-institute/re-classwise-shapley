import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from dvc.api import params_show
from PIL import Image

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


def get_or_create_mlflow_experiment(experiment_name):
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


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--sub-folder", type=str, required=False)
def render_plots(
    experiment_name: str, model_name: str, sub_folder: Optional[str] = None
):
    load_dotenv()
    logger.info("Starting plotting of data valuation experiment")
    setup_plotting()
    experiment_path = Config.RESULT_PATH / experiment_name / model_name
    experiment_name = f"{experiment_name}.{model_name}"
    mlflow.set_tracking_uri("http://localhost:5000")

    if sub_folder is not None:
        experiment_path /= sub_folder
        experiment_name += f".{sub_folder}"

    tmp_dir = Path("tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    experiment_id = get_or_create_mlflow_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        params = params_show()
        params = flatten_dict(params)
        mlflow.log_params(params)

        results_per_dataset = load_results_per_dataset(experiment_path)
        plot_metric_table(results_per_dataset)
        plot_metric_curves(
            results_per_dataset,
            f"Experiment '{experiment_name}' on model '{model_name}'",
        )

    shutil.rmtree(tmp_dir)


def plot_metric_curves(results_per_dataset: Dict, title: str):
    """
    Plot the metric curves for each dataset and valuation method.
    :param results_per_dataset:
    :param title:
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
                ld = len(dataset_results[0].curves[valuation_method_name][metric_name])
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
        mlflow.log_figure(fig, f"curve_{metric_name}.png")


def plot_metric_table(results_per_dataset):
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
                mlflow.log_metric(
                    f"{metric_name}_{dataset_name}_{valuation_method_name}", m
                )

        df_styled = mean_metric.style.highlight_max(color="lightgreen", axis=1)
        output_file = str(f"metrics_{metric_name}.png")
        dfi.export(df_styled, output_file)
        with Image.open(output_file) as im:
            mlflow.log_image(im, output_file)

        os.remove(output_file)


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
