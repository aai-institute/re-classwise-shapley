import os
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
from dvc.api import params_show
from numpy.random import SeedSequence
from pydvl.utils import Dataset, SupervisedModel

from re_classwise_shapley.config import Config
from re_classwise_shapley.experiments import experiment_noise_removal, experiment_wad
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.preprocess import (
    fetch_and_sample_val_test_dataset,
    parse_valuation_methods_config,
)
from re_classwise_shapley.utils import set_random_seed

logger = setup_logger()


@click.command()
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def run_experiment_noise_removal(dataset_name: str, model_name: str):
    """
    Run an experiment and store the results of the run on disk.
    :param dataset_name: Dataset to use.
    :param model_name: Model to use.
    """
    experiment_name = "noise_removal"
    logger.info(f"Starting experiment '{experiment_name}.")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")
    global_settings = params["settings"]
    n_repetitions = global_settings["n_repetitions"]
    del global_settings["n_repetitions"]

    # preprocess valuation methods to be callable from utility to valuation result.
    valuation_methods = params["valuation_methods"]
    valuation_methods_factory = parse_valuation_methods_config(
        valuation_methods, global_settings
    )

    # Create the output directory
    output_dir = Config.RESULT_PATH / experiment_name / model_name

    experiment_seed = abs(int(hash(experiment_name + dataset_name + model_name)))
    seed_sequence = SeedSequence(experiment_seed)
    rng = np.random.default_rng(seed_sequence)
    set_random_seed(rng.integers(2**31 - 1, size=1)[0])
    seeds = seed_sequence.spawn(n_repetitions)

    for repetition in range(n_repetitions):
        logger.info(Config.DOUBLE_BREAK)
        logger.info(f"Executing repetition {repetition} for model '{model_name}'.")

        val_test_set_kwargs = params["datasets"][dataset_name]
        val_test_set = fetch_and_sample_val_test_dataset(
            dataset_name, val_test_set_kwargs, seed=seeds[repetition]
        )

        model_kwargs = params["models"][model_name]
        model = instantiate_model(model_name, model_kwargs)
        repetition_output_dir = output_dir / dataset_name / f"{repetition=}"

        try:
            _run_and_measure_experiment_noise_removal(
                model,
                val_test_set,
                valuation_methods_factory,
                repetition_output_dir,
            )

        except Exception as e:
            logger.error(f"Error while executing experiment '{experiment_name}': {e}")
            logger.info("Skipping experiment and continuing with next one.")
            raise e


def _run_and_measure_experiment_noise_removal(
    model: SupervisedModel,
    val_test_set: Tuple[Dataset, Dataset],
    valuation_methods_factory,
    output_dir: Path,
):
    logger.info(Config.DOUBLE_BREAK)
    logger.info(f"Task 2: Noise removal for classification.")
    logger.info(Config.SINGLE_BREAK)

    results = experiment_noise_removal(
        model=model,
        val_test_set=val_test_set,
        valuation_methods_factory=valuation_methods_factory,
    )
    results.store(output_dir)
    logger.info(f"Results are {results}.")
    logger.info(f"Stored results of experiment in {output_dir}.")


if __name__ == "__main__":
    run_experiment_noise_removal()
