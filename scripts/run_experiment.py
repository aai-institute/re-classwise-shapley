from pathlib import Path

import click
from dvc.api import params_show
from dvc.repo import Repo

from re_classwise_shapley.constants import RANDOM_SEED
from re_classwise_shapley.data.config import Config
from re_classwise_shapley.experiments import experiment_noise_removal, experiment_wad
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.preprocess import (
    parse_datasets_config,
    parse_models_config,
    parse_valuation_methods_config,
)
from re_classwise_shapley.utils import set_random_seed

logger = setup_logger()

set_random_seed(RANDOM_SEED)

DOUBLE_BREAK = 120 * "="
SINGLE_BREAK = 120 * "-"


@click.command()
@click.argument("experiment-name", type=str, nargs=1)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def run_experiment(experiment_name: str, dataset_name: str):
    logger.info("Starting data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")
    global_settings = params["global"]
    n_repetitions = global_settings["n_repetitions"]
    del global_settings["n_repetitions"]

    # preprocess valuation methods to be callable from utility to valuation result.
    valuation_methods = params["valuation_methods"]
    valuation_methods_factory = parse_valuation_methods_config(
        valuation_methods, global_settings
    )

    # preprocess datasets
    datasets_settings = {dataset_name: params["datasets"][dataset_name]}
    datasets = parse_datasets_config(datasets_settings)

    # preprocess models_config
    models_config = params["models"]
    model_generator_factory = parse_models_config(models_config)

    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root()) / Config.RESULT_PATH / experiment_name / dataset_name
    )

    for model_name in models_config.keys():
        for repetition in range(n_repetitions):
            logger.info(DOUBLE_BREAK)
            logger.info(f"Executing repetition {repetition} for model '{model_name}'.")
            experiments_output_dir = (
                experiment_output_dir / model_name / f"{repetition=}"
            )

            try:
                # TODO: Yeah, this is a bit hacky. But it works for now.
                match experiment_name:
                    case "wad_drop":
                        _run_and_measure_experiment_wad_drop(
                            datasets,
                            model_name,
                            model_generator_factory,
                            valuation_methods_factory,
                            experiments_output_dir,
                        )

                    case "noise_removal":
                        _run_and_measure_experiment_noise_removal(
                            datasets,
                            model_name,
                            model_generator_factory,
                            valuation_methods_factory,
                            experiments_output_dir,
                        )

                    case "wad_drop_transfer":
                        experiment_three_settings = params["experiments"][
                            "wad_drop_transfer"
                        ]
                        experiment_three_path = experiments_output_dir

                        # preprocess models_config
                        test_models_config = params["experiments"]["wad_drop_transfer"][
                            "transfer_models"
                        ]
                        test_model_generator_factory = parse_models_config(
                            test_models_config
                        )

                        for test_model_name, _ in experiment_three_settings[
                            "transfer_models"
                        ].items():
                            if test_model_name != model_name:
                                _run_and_measure_experiment_wad_drop_transfer(
                                    datasets,
                                    model_name,
                                    test_model_name,
                                    model_generator_factory,
                                    test_model_generator_factory,
                                    valuation_methods_factory,
                                    experiment_three_path,
                                )

                    case _:
                        raise NotImplementedError(
                            f"The experiment '{experiment_name}' is not implemented."
                        )

            except Exception as e:
                logger.error(
                    f"Error while executing experiment '{experiment_name}': {e}"
                )
                logger.info("Skipping experiment and continuing with next one.")
                raise e


def _run_and_measure_experiment_wad_drop(
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
    experiment_one_path = experiments_output_dir
    results = experiment_wad(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
    )
    results.store(experiment_one_path)
    logger.info(f"Results are {results}.")
    logger.info(f"Stored results of experiment one in {experiment_one_path}.")


def _run_and_measure_experiment_noise_removal(
    datasets,
    model_name,
    model_generator_factory,
    valuation_methods_factory,
    experiments_output_dir,
):
    logger.info(DOUBLE_BREAK)
    logger.info(f"Calculating task 2: Noise removal for classification.")
    model = model_generator_factory[model_name]()
    experiment_two_path = experiments_output_dir
    results = experiment_noise_removal(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
    )
    results.store(experiment_two_path)
    logger.info(f"Results are {results}.")
    logger.info(f"Stored results of experiment two in {experiment_two_path}.")


def _run_and_measure_experiment_wad_drop_transfer(
    datasets,
    model_name,
    test_model_name,
    model_generator_factory,
    test_model_generator_factory,
    valuation_methods_factory,
    experiment_three_path,
):
    logger.info(DOUBLE_BREAK)
    logger.info(
        f"Calculating task 3: Value transfer to dfifferent model implementations."
    )
    logger.info(f"Testing against model '{test_model_name}'.")
    model = model_generator_factory[model_name]()
    test_model = test_model_generator_factory[test_model_name]()
    experiment_three_test_path = experiment_three_path / test_model_name
    results = experiment_wad(
        model=model,
        datasets=datasets,
        valuation_methods_factory=valuation_methods_factory,
        test_model=test_model,
    )
    results.store(experiment_three_test_path)
    logger.info(f"Results are {results}.")
    logger.info(
        f"Stored results of experiment three against '{test_model_name}' in {experiment_three_test_path}."
    )


if __name__ == "__main__":
    run_experiment()
