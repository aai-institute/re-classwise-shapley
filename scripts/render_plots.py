import os
import re
from pathlib import Path

import click
import pandas as pd
from dvc.api import params_show

from csshapley22.constants import OUTPUT_DIR, RANDOM_SEED
from csshapley22.log import setup_logger
from csshapley22.plotting import (
    plot_utility_over_num_removals,
    plot_values_histogram,
    setup_plotting,
)
from csshapley22.utils import set_random_seed

logger = setup_logger()

setup_plotting()
set_random_seed(RANDOM_SEED)


@click.command()
@click.argument("experiment-name", type=str, nargs=1)
@click.option("--dataset-name", type=str, required=True)
def render_plots(experiment_name: str, dataset_name: str):
    logger.info("Starting plotting of data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")

    experiment_input_dir = OUTPUT_DIR / "results" / experiment_name / dataset_name
    plots_output_dir = OUTPUT_DIR / "plots" / experiment_name / dataset_name
    for model_name in os.listdir(experiment_input_dir):
        model_input_dir = experiment_input_dir / model_name
        model_plots_output_dir = plots_output_dir / model_name
        os.makedirs(model_plots_output_dir, exist_ok=True)

        if experiment_name == "wad_drop":
            metrics, valuation_results, scores = load_results(
                model_input_dir,
                load_scores=True,
            )
            plot_utility_over_num_removals(
                scores,
                output_dir=model_plots_output_dir,
            )
            plot_values_histogram(
                valuation_results,
                output_dir=model_plots_output_dir,
            )
        elif experiment_name == "noise_removal":
            metrics, valuation_results = load_results(
                model_input_dir,
            )
            plot_values_histogram(
                valuation_results,
                output_dir=model_plots_output_dir,
            )

        elif experiment_name == "wad_drop_transfer":
            for sub_folder in os.listdir(model_input_dir / "repetition=0"):
                metrics, valuation_results, scores = load_results(
                    model_input_dir,
                    load_scores=True,
                    sub_folder=sub_folder,
                )
                sub_plots_output_dir = model_plots_output_dir / sub_folder
                os.makedirs(sub_plots_output_dir, exist_ok=True)
                plot_utility_over_num_removals(
                    scores,
                    output_dir=sub_plots_output_dir,
                )
                plot_values_histogram(
                    valuation_results,
                    output_dir=sub_plots_output_dir,
                )

        logger.info("Finished data valuation experiment")


def load_results(
    model_input_dir: Path, *, sub_folder: str = None, load_scores: bool = False
):
    metrics = None
    valuation_results = None
    scores = None

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
            it_scores = pd.read_pickle(repetition_input_dir / "scores.pkl")
            scores = (
                it_metrics
                if metrics is None
                else pd.concat((scores, it_scores), axis=0)
            )
    metrics = metrics.reset_index(drop=True)
    valuation_results = valuation_results.reset_index(drop=True).applymap(
        lambda x: x.values
    )

    if load_scores:
        scores = scores.reset_index(drop=True)
        return metrics, valuation_results, scores
    else:
        return metrics, valuation_results


if __name__ == "__main__":
    render_plots()
