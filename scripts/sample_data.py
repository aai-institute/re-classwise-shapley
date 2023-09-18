import json
import os
import pickle
from functools import partial
from typing import Dict, List, Tuple

import click
import numpy as np
from dvc.api import params_show
from numpy.random import SeedSequence
from numpy.typing import NDArray
from pydvl.utils import Dataset, ensure_seed_sequence

from re_classwise_shapley.config import Config
from re_classwise_shapley.io import load_dataset
from re_classwise_shapley.preprocess import flip_labels
from re_classwise_shapley.types import Seed
from re_classwise_shapley.utils import get_pipeline_seed

PreprocessorRegistry = {"flip_labels": flip_labels}


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
def sample_data(
    experiment_name: str,
    dataset_name: str,
    repetition_id: int,
):
    """
    Samples a dataset from a preprocessed dataset. It accepts `experiment_name` and
    `dataset_name` as arguments. The experiment name is used to determine the sampling
    method. The dataset name is used to determine the dataset to sample from. First it

    Args:
        experiment_name: Name of the executed experiment. Experiments define the
            sampling method.
        dataset_name: Dataset to use.
        repetition_id: Repetition id of the experiment. It is used also as a seed for
            all randomness.
    """
    params = params_show()
    output_dir = (
        Config.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    os.makedirs(output_dir, exist_ok=True)

    experiment_config = params["experiments"][experiment_name]
    sampler_name = experiment_config["sampler"]
    sampler_kwargs = params["samplers"][sampler_name]

    seed = get_pipeline_seed(repetition_id, 1)
    seed_sequence = SeedSequence(seed).spawn(2)
    val_set, test_set = sample_val_test_set(
        dataset_name, sampler_kwargs, seed=seed_sequence[0]
    )

    preprocess_info = None
    if "preprocessors" in experiment_config:
        preprocess_info, val_set = apply_preprocessors(
            val_set, experiment_config["preprocessors"], seed_sequence
        )

    with open(output_dir / "val_set.pkl", "wb") as file:
        pickle.dump(val_set, file)

    with open(output_dir / "test_set.pkl", "wb") as file:
        pickle.dump(test_set, file)

    if preprocess_info and len(preprocess_info) > 0:
        with open(output_dir / "preprocess_info.json", "w") as file:
            json.dump(preprocess_info, file, indent=4, sort_keys=True)


def apply_preprocessors(val_set: Dataset, preprocessor_configs: Dict, seed: List[Seed]):
    preprocess_info = {}
    for idx, (preprocessor_name, preprocessor_config) in enumerate(
        preprocessor_configs.items()
    ):
        preprocessor_fn = PreprocessorRegistry[preprocessor_name]
        val_set, info = preprocessor_fn(val_set, **preprocessor_config, seed=seed[idx])
        preprocess_info.update(
            {f"preprocessor.{preprocessor_name}.{k}": v for k, v in info.items()}
        )

    return val_set, preprocess_info


def sample_val_test_set(
    dataset_name: str, sample_info: Dict, max_samples: int = None, seed: Seed = None
) -> Tuple[Dataset, Dataset]:
    """
    Takes the name of a (pre-processed) dataset and sampling information and samples the
    validation and test set. The sampling information is expected to be a dictionary
    with fields `train`, `val` and `test` which contain the relative percentages of the
    respective set.

    Args:
        dataset_name: Name of the dataset.
        max_samples field: Limit the number of samples taken from the dataset.
        seed: Either an instance of a numpy random number generator or a seed for it.
    Returns:
        Tuple containing the validation and test set.
    """
    rng = np.random.default_rng(seed)

    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_name
    x, y, _ = load_dataset(preprocessed_folder)
    p = rng.permutation(len(x))
    x, y = x[p], y[p]

    n_samples = min(sample_info["max_samples"], len(x))
    val_size = int(n_samples * sample_info["val"])
    train_size = int(n_samples * sample_info["train"])
    test_size = int(n_samples * sample_info["test"])
    set_train, set_val, set_test = stratified_sampling(
        x, y, (train_size, val_size, test_size), seed=rng
    )
    validation_set = Dataset(*set_train, *set_val)
    test_set = Dataset(*set_train, *set_test)
    return validation_set, test_set


def stratified_sampling(
    features: NDArray[np.float_],
    labels: NDArray[np.int_],
    sizes: Tuple[int, ...],
    *,
    seed: Seed = None,
) -> Tuple[Tuple[NDArray[np.float_], NDArray[np.int_]], ...]:
    """
    Stratified sampler of a dataset containing features and labels.

    :param features: Features of the dataset.
    :param labels: Labels of the features.
    :param sizes: Tuple containing target size of each dataset.
    :param seed: Seed for the random number generator.
    :return: A tuple containing tuples of the sub-sampled data.
    """
    if len(features) != len(labels):
        raise ValueError("The number of features and labels must be equal.")

    if np.sum(sizes) > len(features):
        raise ValueError("The sum of the sizes exceeds the size of the dataset.")

    rng = np.random.default_rng(seed)
    p = rng.permutation(len(features))
    features = features[p]
    labels = labels[p]
    unique_labels, num_labels = np.unique(labels, return_counts=True)
    label_indices = [np.argwhere(labels == label)[:, 0] for label in unique_labels]
    relative_set_sizes = num_labels / len(labels)

    data = [list() for _ in sizes]
    it_idx = [0 for _ in unique_labels]

    for i, size in enumerate(sizes):
        absolute_set_sizes = (relative_set_sizes * size).astype(np.int_)
        missing_elements = size - np.sum(absolute_set_sizes)
        absolute_set_sizes[np.argsort(absolute_set_sizes)[:missing_elements]] += 1

        if np.sum(absolute_set_sizes) != size:
            raise ValueError("There is an error in sampling.")

        for j in range(len(unique_labels)):
            label_idx = label_indices[j]
            window_idx = label_idx[it_idx[j] : it_idx[j] + absolute_set_sizes[j]]
            it_idx[j] += absolute_set_sizes[j]
            data[i].append((features[window_idx], labels[window_idx]))

    data = [
        (np.concatenate([t[0] for t in lst]), np.concatenate([t[1] for t in lst]))
        for lst in data
    ]
    return tuple(data)


if __name__ == "__main__":
    sample_data()
