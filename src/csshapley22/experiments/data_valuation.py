from functools import partial
from pathlib import Path

import click
from dvc.api import params_show
from dvc.repo import Repo

from csshapley22.constants import RANDOM_SEED
from csshapley22.dataset import DatasetRegistry
from csshapley22.experiments.experiment_one import run_experiment_one
from csshapley22.utils import set_random_seed, setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()

set_random_seed(RANDOM_SEED)


@click.command()
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
def run(model_name: str):
    logger.info("Starting data valuation experiment")

    params = params_show()["experiment_one"]
    logger.info(f"Using parameters:\n{params}")

    valuation_methods = params["valuation_methods"]
    datasets = params["datasets"]

    datasets = {
        dataset_name: DatasetRegistry[dataset_name](**dataset_kwargs)
        for dataset_name, dataset_kwargs in datasets.items()
    }
    valuation_functions = {
        valuation_method_name: partial(
            compute_values, valuation_method=valuation_method_name
        )
        for valuation_method_name in valuation_methods.keys()
    }
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

        result = run_experiment_one(
            model_name=model_name,
            datasets=datasets,
            valuation_functions=valuation_functions,
        )
        logger.info("Saving results to disk")
        result.metric.to_csv(
            repetition_output_dir / "weighted_accuracy_drops.csv", index=False
        )

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    run()
