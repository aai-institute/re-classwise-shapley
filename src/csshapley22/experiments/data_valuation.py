from datetime import datetime
from pathlib import Path

import click
from dvc.api import params_show
from dvc.repo import Repo

from csshapley22.constants import RANDOM_SEED
from csshapley22.experiments.all import experiment_noise_removal, experiment_wad
from csshapley22.preprocess import (
    parse_datasets_config,
    parse_models_config,
    parse_valuation_methods_config,
)
from csshapley22.utils import instantiate_model, set_random_seed, setup_logger

logger = setup_logger()

set_random_seed(RANDOM_SEED)


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

    # preprocess models
    models = general_settings["models"]
    model_generator_factory = parse_models_config(models)

    n_repetitions = general_settings["n_repetitions"]
    date_iso_format = datetime.now().isoformat()

    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root()) / "output" / "results" / date_iso_format
    )

    for repetition in range(n_repetitions):
        logger.info(f"{repetition=}")
        repetition_output_dir = experiment_output_dir / f"{repetition=}"

        for model_name in models.keys():
            # Experiment One
            model = model_generator_factory[model_name]()
            experiment_one_path = repetition_output_dir / "wad"
            experiment_wad(
                model=model,
                datasets=datasets,
                valuation_methods_factory=valuation_methods_factory,
            ).store(experiment_one_path)

            # Experiment Two
            model = model_generator_factory[model_name]()
            experiment_two_path = repetition_output_dir / "noise_removal"
            experiment_noise_removal(
                model=model,
                datasets=datasets,
                valuation_methods_factory=valuation_methods_factory,
            ).store(experiment_two_path)

            # Experiment Three
            model = model_generator_factory[model_name]()
            experiment_three_settings = params["experiment_3"]
            experiment_three_path = repetition_output_dir / "value_transfer"

            for test_model_name in experiment_three_settings["test_models"]:
                if test_model_name not in models:
                    raise ValueError(f"The model '{test_model_name}' doesn't exist.")

                test_model = model_generator_factory[test_model_name]()
                experiment_three_test_path = experiment_three_path / test_model_name
                experiment_wad(
                    model=model,
                    datasets=datasets,
                    valuation_methods_factory=valuation_methods_factory,
                    test_model=test_model,
                ).store(experiment_three_test_path)

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    run()
