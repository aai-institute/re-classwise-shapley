from functools import partial
from pathlib import Path

import click
from dvc.api import params_show
from dvc.repo import Repo

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.fetch import fetch_datasets
from csshapley22.experiments.experiment_one import run_experiment_one
from csshapley22.experiments.experiment_two import run_experiment_two
from csshapley22.utils import set_random_seed, setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()

set_random_seed(RANDOM_SEED)


@click.command()
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
def run(model_name: str):
    logger.info("Starting data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")
    valuation_methods = params["settings"]["valuation_methods"]

    datasets = fetch_datasets()
    n_repetitions = params["settings"]["n_repetitions"]

    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root()) / "output" / "data_valuation" / "results"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(n_repetitions):
        logger.info(f"{repetition=}")

        repetition_output_dir = experiment_output_dir / f"{repetition=}"
        repetition_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_experiment_two(
            model_name=model_name,
            datasets=datasets,
            valuation_methods=valuation_methods,
        )

        result = run_experiment_one(
            model_name=model_name,
            datasets=datasets,
            valuation_methods=valuation_methods,
        )
        logger.info("Saving results to disk")
        result.metric.to_csv(repetition_output_dir / "weighted_accuracy_drops.csv")

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    run()
