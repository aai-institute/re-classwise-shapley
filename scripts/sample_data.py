"""
Stage 3 for sampling data from a preprocessed dataset.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

The sampling method is described by the experiment. Each experiment has a sampler
defined in the `samplers` section of the `params.yaml` file. Furthermore, preprocessors
are defined per experiment. For example one can randomly flip a percentage of the labels
in the dataset.
"""

import click
from numpy.random import SeedSequence

from re_classwise_shapley.io import (
    Accessor,
    has_val_test_dataset,
    load_raw_dataset,
    store_val_test_data,
)
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.preprocess import apply_sample_preprocessors
from re_classwise_shapley.rand import sample_val_test_set
from re_classwise_shapley.utils import load_params_fast, pipeline_seed

logger = setup_logger("sample_data")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
def sample_data(
    experiment_name: str,
    dataset_name: str,
):
    """
    Samples a dataset from a preprocessed dataset. It accepts `experiment_name` and
    `dataset_name` as arguments. The experiment name is used to determine the sampling
    method. The dataset name is used to determine the dataset to sample from. The
    repetition id is used as a seed for all randomness. The sampled dataset is stored in
    the `Accessor.SAMPLED_PATH` directory.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
    """
    _sample_data(experiment_name, dataset_name)


def _sample_data(
    experiment_name: str,
    dataset_name: str,
):
    params = load_params_fast()
    input_folder = Accessor.PREPROCESSED_PATH / dataset_name
    dataset_folder = Accessor.SAMPLED_PATH / experiment_name / dataset_name
    if has_val_test_dataset(dataset_folder):
        return logger.info(f"Sampled data exists in '{dataset_folder}'. Skipping...")

    logger.info(f"Sampling data for '{experiment_name}/{dataset_name}'")
    n_pipeline_step = 3
    seed = pipeline_seed(42, n_pipeline_step)
    seed_sequence = SeedSequence(seed).spawn(2)

    experiment_config = params["experiments"][experiment_name]
    sampler_name = experiment_config["sampler"]
    sampler_kwargs = params["samplers"][sampler_name]

    logger.info(f"Loading raw data. ")
    x, y, _ = load_raw_dataset(input_folder)
    val_set, test_set = sample_val_test_set(
        x, y, **sampler_kwargs, seed=seed_sequence[0]
    )

    logger.info(f"Apply preprocessors.")
    preprocess_info = None
    if "preprocessors" in experiment_config:
        val_set, preprocess_info = apply_sample_preprocessors(
            val_set, experiment_config["preprocessors"], seed_sequence
        )

    logger.info(f"Store results.")
    store_val_test_data(val_set, test_set, preprocess_info, dataset_folder)


if __name__ == "__main__":
    sample_data()
