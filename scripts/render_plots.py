import json
import os
import re
from pathlib import Path

import click
import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dvc.api import params_show
from pandas.plotting import table

from re_classwise_shapley.constants import OUTPUT_DIR, RANDOM_SEED
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import (
    plot_curve,
    plot_values_histogram,
    setup_plotting,
)
from re_classwise_shapley.utils import set_random_seed

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


@click.command()
@click.argument("experiment-name", type=str, nargs=1)
def render_plots(experiment_name: str):
    logger.info("Starting plotting of data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")
    plot_order = [
        "cifar10",
        "click",
        "covertype",
        "cpu",
        "diabetes",
        "fmnist_binary",
        "mnist_binary",
        "mnist_multi",
        "phoneme",
    ]
    plot_ax_index = {col: idx for idx, col in enumerate(plot_order)}
    figsize = (20, 6)

    experiment_input_dir = OUTPUT_DIR / "results" / experiment_name / plot_order[0]
    model_input_dir = experiment_input_dir / "logistic_regression"
    metrics, valuation_results, curves = load_results(
        model_input_dir,
        load_scores=True,
    )
    plt_axes = {}
    all_keys = curves.iloc[0, 0].keys()

    for key in all_keys:
        fig, ax = plt.subplots(2, 5, figsize=figsize)
        ax = np.array(ax).flatten()
        plt_axes[key] = (fig, ax)

    for method_name in valuation_results.columns:
        fig, ax = plt.subplots(2, 5, figsize=figsize)
        ax = np.array(ax).flatten()
        plt_axes[f"histogram_{method_name}"] = (fig, ax)

    for model_name in params["models"].keys():
        plots_output_dir = OUTPUT_DIR / "plots" / experiment_name / model_name
        key_metrics = {key: {} for key in all_keys}

        for dataset_name in params["datasets"].keys():
            dataset_index = plot_ax_index[dataset_name]
            experiment_input_dir = (
                OUTPUT_DIR / "results" / experiment_name / dataset_name
            )
            model_input_dir = experiment_input_dir / model_name
            os.makedirs(plots_output_dir, exist_ok=True)
            dataset_letter = chr(ord("`") + dataset_index + 1)
            metrics, valuation_results, curves = load_results(
                model_input_dir,
                load_scores=True,
            )

            for key in all_keys:
                key_metrics[key][dataset_name] = metrics.applymap(
                    lambda v: v[key]
                ).mean(axis=0)

            if experiment_name == "wad_drop":
                if not isinstance(curves.iloc[0, 0], dict):
                    curves = curves.applymap(lambda s: {"highest_wad_drop": s})

                for key in curves.iloc[0, 0].keys():
                    fig, ax = plt_axes[key]
                    plot_curve(
                        curves.applymap(lambda s: [s[key]]).applymap(lambda x: x[0]),
                        title=f"({dataset_letter}) {dataset_name}",
                        ax=ax[dataset_index],
                    )
                    fig.suptitle(
                        f"Experiment 1: Weighted-accuracy-drop (WAD) usign {key} for model '{model_name}'"
                    )

                for method_name in valuation_results.columns:
                    fig, ax = plt_axes[f"histogram_{method_name}"]
                    plot_values_histogram(
                        valuation_results.loc[:, method_name],
                        title=f"({dataset_letter}) {dataset_name}",
                        ax=ax[dataset_index],
                    )
                    fig.suptitle(
                        f"Experiment 1: Histogram for model '{model_name}' and method '{method_name}'"
                    )

        key_metrics = {
            key: pd.DataFrame(key_metrics[key]).T for key in key_metrics.keys()
        }
        for key in key_metrics.keys():
            df = key_metrics[key]
            df_styled = df.style.highlight_max(color="lightgreen", axis=1)
            dfi.export(df_styled, plots_output_dir / f"metrics_{key}.png")

        for key, (fig, ax) in plt_axes.items():
            fig.subplots_adjust(hspace=0.3)
            fig.savefig(plots_output_dir / f"{key}.png")

        logger.info("Finished plotting.")


def load_results(
    model_input_dir: Path, *, sub_folder: str = None, load_scores: bool = False
):
    metrics = None
    valuation_results = None
    curves = None

    for repetition in os.listdir(model_input_dir):
        repetition_input_dir = model_input_dir / repetition
        if sub_folder is not None:
            repetition_input_dir = repetition_input_dir / sub_folder

        it_metrics = pd.read_csv(repetition_input_dir / "metric.csv").iloc[:, 1:]
        it_valuation_results = pd.read_pickle(
            repetition_input_dir / "valuation_results.pkl"
        )

        metrics = (
            it_metrics if metrics is None else pd.concat((metrics, it_metrics), axis=0)
        )
        valuation_results = (
            it_valuation_results
            if valuation_results is None
            else pd.concat((valuation_results, it_valuation_results), axis=0)
        )

        if load_scores:
            it_curves = pd.read_pickle(repetition_input_dir / "curves.pkl")
            curves = (
                it_curves if curves is None else pd.concat((curves, it_curves), axis=0)
            )
    metrics = metrics.reset_index(drop=True)
    valuation_results = valuation_results.reset_index(drop=True).applymap(
        lambda x: x.values
    )
    metrics = metrics.applymap(lambda v: json.loads(v.replace("'", '"')))

    if load_scores:
        curves = curves.reset_index(drop=True)
        return metrics, valuation_results, curves
    else:
        return metrics, valuation_results


if __name__ == "__main__":
    render_plots()
