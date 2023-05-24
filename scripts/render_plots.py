import os
import re

import click
import pandas as pd
from dvc.api import params_show

from csshapley22.constants import OUTPUT_DIR, RANDOM_SEED
from csshapley22.log import setup_logger
from csshapley22.plotting import (
    plot_utility_over_removal_percentages,
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

    data_valuation_params = params["data_valuation"]
    removal_percentages = data_valuation_params["removal_percentages"]
    method_names = data_valuation_params["method_names"]

    experiment_output_dir = OUTPUT_DIR / "data_valuation" / "results"
    assert experiment_output_dir.exists()
    plots_output_dir = OUTPUT_DIR / "data_valuation" / "plots"
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    budget_regex = re.compile(r"budget=(?P<budget>\d+)/")

    all_values = []
    for file in experiment_output_dir.rglob("values*.csv"):
        df = pd.read_csv(file)
        budget = re.search(budget_regex, os.fspath(file)).group("budget")
        df["budget"] = budget
        all_values.append(df)
    values_df = pd.concat(all_values)

    all_scores = []
    for file in experiment_output_dir.rglob("scores*.csv"):
        df = pd.read_csv(file)
        budget = re.search(budget_regex, os.fspath(file)).group("budget")
        df["budget"] = budget
        all_scores.append(df)
    scores_df = pd.concat(all_scores)

    plot_utility_over_removal_percentages(
        scores_df,
        budgets=scores_df["budget"].unique(),
        method_names=method_names,
        removal_percentages=removal_percentages,
        output_dir=plots_output_dir,
    )

    plot_values_histogram(
        values_df,
        hue_column="budget",
        method_names=method_names,
        output_dir=plots_output_dir,
    )

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    render_plots()
