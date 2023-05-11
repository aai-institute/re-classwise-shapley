import logging
from datetime import datetime
from pathlib import Path

import click
from dvc.api import params_show
from dvc.repo import Repo

from csshapley22.constants import RANDOM_SEED
from csshapley22.log import setup_logger
from csshapley22.preprocess import (
    parse_datasets_config,
    parse_models_config,
    parse_valuation_methods_config,
)
from csshapley22.stages.all import experiment_noise_removal, experiment_wad
from csshapley22.utils import set_random_seed, timeout

logger = setup_logger()

set_random_seed(RANDOM_SEED)

DOUBLE_BREAK = 120 * "="
SINGLE_BREAK = 120 * "-"


@click.command()
def run():
    logger.info("Starting data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")
    general_settings = params["general"]

    # preprocess valuation methods to be callable from utility to valuation result.
    valuation_methods = general_settings["valuation_methods"]
    valuation_methods_factory = parse_valuation_methods_config(valuation_methods)

    # preprocess datasets
    datasets_settings = general_settings["datasets"]
    datasets = parse_datasets_config(datasets_settings)

    # preprocess models_config
    models_config = general_settings["models"]
    model_generator_factory = parse_models_config(models_config)

    n_repetitions = general_settings["n_repetitions"]

    # Create the output directory
    experiment_output_dir = Path(Repo.find_root()) / "output" / "results"

    for repetition in range(n_repetitions):
        logger.info(DOUBLE_BREAK)
        repetition_output_dir = experiment_output_dir / f"{repetition=}"

        for model_name in models_config.keys():
            logger.info(f"Executing repetition {repetition} for model '{model_name}'.")
            experiments_output_dir = repetition_output_dir / model_name

            _run_and_measure_experiment_one(
                datasets,
                model_name,
                model_generator_factory,
                valuation_methods_factory,
                experiments_output_dir,
            )

            _run_and_measure_experiment_two(
                datasets,
                model_name,
                model_generator_factory,
                valuation_methods_factory,
                experiments_output_dir,
            )

            experiment_three_settings = params["experiment_3"]
            experiment_three_path = experiments_output_dir / "value_transfer"

            for test_model_name in experiment_three_settings["test_models"]:
                _run_and_measure_experiment_three(
                    datasets,
                    model_name,
                    test_model_name,
                    model_generator_factory,
                    valuation_methods_factory,
                    experiment_three_path,
                )

    logger.info("Finished data valuation experiment")


def _run_and_measure_experiment_one(
    datasets,
    model_name,
    model_generator_factory,
    valuation_methods_factory,
    experiments_output_dir,
):
    logger.info(SINGLE_BREAK)
    logger.info(f"Task 1: Weighted absolute difference.")
    logger.info(SINGLE_BREAK)

    model = model_generator_factory[model_name]()
    experiment_one_path = experiments_output_dir / "wad"
    results = experiment_wad(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
    )
    results.store(experiment_one_path)
    logger.info(f"Results are {results}.")
    logger.info("Stored results of experiment one.")


def _run_and_measure_experiment_two(
    datasets,
    model_name,
    model_generator_factory,
    valuation_methods_factory,
    experiments_output_dir,
):
    logger.info(DOUBLE_BREAK)
    logger.info(f"Calculating task 2: Noise removal for classification.")
    model = model_generator_factory[model_name]()
    experiment_two_path = experiments_output_dir / "noise_removal"
    results = experiment_noise_removal(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
    )
    results.store(experiment_two_path)
    logger.info(f"Results are {results}.")
    logger.info("Stored results of experiment two.")


def _run_and_measure_experiment_three(
    datasets,
    model_name,
    test_model_name,
    model_generator_factory,
    valuation_methods_factory,
    experiment_three_path,
):
    logger.info(DOUBLE_BREAK)
    logger.info(
        f"Calculating task 3: Value transfer to dfifferent model implementations."
    )
    logger.info(f"Testing against model '{test_model_name}'.")
    model = model_generator_factory[model_name]()
    test_model = model_generator_factory[test_model_name]()
    experiment_three_test_path = experiment_three_path / test_model_name
    results = experiment_wad(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
        test_model=test_model,
    )
    results.store(experiment_three_test_path)
    logger.info(f"Results are {results}.")
    logger.info(f"Stored results of experiment three against'{test_model_name}'.")


if __name__ == "__main__":
    run()
