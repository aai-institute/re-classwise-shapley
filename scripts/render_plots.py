import os
import os.path
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dotenv import load_dotenv
from dvc.api import params_show
from matplotlib.colors import to_rgb
from PIL import Image
from plotly.subplots import make_subplots

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
    "owen_sampling_shapley": "purple",
    "banzhaf_shapley": "turquoise",
    "least_core": "gray",
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


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


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

    logger.info("Starting run.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        # TODO Uncomment
        # logger.info("Log params.")
        # params = params_show()
        # params = flatten_dict(params)
        # mlflow.log_params(params)<
        #
        # logger.info("Packing results.")
        # filename = f"results"
        # shutil.make_archive(filename, "tar", experiment_path)
        # logger.info("Uploading results.")
        # tar_filename = f"{filename}.tar"
        # mlflow.log_artifact(tar_filename)
        # os.remove(tar_filename)

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
        if metric_name == "time":
            continue

        logger.info(f"Plotting metric {metric_name}.")
        fig = make_subplots(rows=h, cols=w, subplot_titles=dataset_names)

        idx = 0
        for row in range(1, h + 1):
            for col in range(1, w + 1):
                if idx >= num_datasets:
                    break

                dataset_name = dataset_names[idx]
                logger.info(f"Plotting dataset {dataset_name}.")

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

                plot_curve(d, fig, row, col)
                idx += 1

        for idx, axis in enumerate(fig.layout):
            if idx != 9 and "xaxis" in axis:
                fig.layout[axis].update(matches="x")

        # # Add common x-label
        # fig.add_annotation(
        #     dict(
        #         x=0.5,
        #         y=-0.15,
        #         showarrow=False,
        #         text="Common X-label",
        #         xref="paper",
        #         yref="paper",
        #         font=dict(size=16)
        #     )
        # )
        #
        # # Add common y-label
        # fig.add_annotation(
        #     dict(
        #         x=-0.15,
        #         y=0.5,
        #         showarrow=False,
        #         text="Common Y-label",
        #         textangle=-90,
        #         xref="paper",
        #         yref="paper",
        #         font=dict(size=16)
        #     )
        # )

        fig.update_layout(title_text=f"{title} with metric '{metric_name}'")
        mlflow.log_figure(fig, f"curve_{metric_name}.html")


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
