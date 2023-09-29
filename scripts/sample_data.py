import json
import os
import pickle
from typing import Dict, List, Tuple

import click
import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray
from pydvl.utils import Dataset

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.io import load_dataset
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import Seed
from re_classwise_shapley.utils import load_params_fast, pipeline_seed

logger = setup_logger("sample_data")


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
    method. The dataset name is used to determine the dataset to sample from. The
    repetition id is used as a seed for all randomness. The sampled dataset is stored in
    the `Accessor.SAMPLED_PATH` directory.

    Args:
        experiment_name: Name of the executed experiment. Experiments define the
            sampling method.
        dataset_name: Dataset to use.
        repetition_id: Repetition id of the experiment. It is used also as a seed for
            all randomness.
    """
    params = load_params_fast()
    input_folder = Accessor.PREPROCESSED_PATH / dataset_name
    output_dir = (
        Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    if os.path.exists(output_dir / "val_set.pkl") and os.path.exists(
        output_dir / "test_set.pkl"
    ):
        return logger.info(f"Sampled data exists in '{output_dir}'. Skipping...")

    seed = pipeline_seed(repetition_id, 1)
    seed_sequence = SeedSequence(seed).spawn(2)

    experiment_config = params["experiments"][experiment_name]
    sampler_name = experiment_config["sampler"]
    sampler_kwargs = params["samplers"][sampler_name]

    x, y, _ = load_dataset(input_folder)
    val_set, test_set = sample_val_test_set(
        x, y, **sampler_kwargs, seed=seed_sequence[0]
    )

    preprocess_info = None
    if "preprocessors" in experiment_config:
        val_set, preprocess_info = apply_preprocessors(
            val_set, experiment_config["preprocessors"], seed_sequence
        )

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / "val_set.pkl", "wb") as file:
        pickle.dump(val_set, file)

    with open(output_dir / "test_set.pkl", "wb") as file:
        pickle.dump(test_set, file)

    if preprocess_info and len(preprocess_info) > 0:
        with open(output_dir / "preprocess_info.json", "w") as file:
            json.dump(preprocess_info, file, indent=4, sort_keys=True)


def sample_val_test_set(
    features: NDArray[np.float_],
    labels: NDArray[np.int_],
    val: float,
    train: float,
    test: float,
    max_samples: int,
    seed: Seed = None,
) -> Tuple[Dataset, Dataset]:
    """
    Takes the name of a (pre-processed) dataset and sampling information and samples the
    validation and test set. Both validation and test set share the same training set.
    All three sets are sampled from the same dataset while preserving the relative label
    distribution of the original dataset by using stratified sampling.

    Args:
        features: Features of the dataset.
        labels: Labels of the dataset.
        val: Relative size of the validation set in percent.
        train: Relative size of the training set in percent.
        test: Relative size of the test set in percent.
        max_samples: Limit the number of samples taken from the dataset.
        seed: Either an instance of a numpy random number generator or a seed for it.
    Returns:
        Tuple containing the validation and test set. Both share the same training set.
    """
    rng = np.random.default_rng(seed)

    p = rng.permutation(len(features))
    features, labels = features[p], labels[p]

    n_samples = min(max_samples, len(features))
    val_size = int(n_samples * val)
    train_size = int(n_samples * train)
    test_size = int(n_samples * test)
    set_train, set_val, set_test = stratified_sampling(
        features, labels, (train_size, val_size, test_size), seed=rng
    )
    validation_set = Dataset(*set_train, *set_val)
    test_set = Dataset(*set_train, *set_test)
    return validation_set, test_set


def stratified_sampling(
    features: NDArray[np.float_],
    labels: NDArray[np.int_],
    n_samples: Tuple[int, ...],
    *,
    seed: Seed = None,
) -> Tuple[Tuple[NDArray[np.float_], NDArray[np.int_]], ...]:
    """
    Stratified sampler of a dataset containing features and labels. The sampler ensures
    that the relative class distribution is preserved in the sampled data. The sampler
    also ensures that for each value of `n_samples` a dataset with exactly `n_samples`
    is returned.

    Args:
        features: Features of the dataset.
        labels: Labels of the features.
        n_samples: Tuple containing number of samples for each dataset.
        seed: Seed for the random number generator.

    Returns:
        A tuple containing of tuples of features and labels. Each inner tuple contains
        the features and labels of one dataset as specified by `n_samples`.
    """
    n_data_points = len(features)
    if n_data_points != len(labels):
        raise ValueError("Labels have to be of same size as features.")

    if np.sum(n_samples) > n_data_points:
        raise ValueError(
            f"The sum of all required samples exceeds `{n_data_points}` available data "
            f"points."
        )

    rng = np.random.default_rng(seed)
    p = rng.permutation(n_data_points)
    features, labels = features[p], labels[p]

    unique_labels, counts = np.unique(labels, return_counts=True)
    relative_set_sizes = counts / len(labels)
    windows = [np.where(labels == label)[0] for label in unique_labels]
    windows_idx = np.zeros(len(windows), dtype=np.int_)
    sampled_data = []

    for size in n_samples:
        sampled_features, sampled_labels = [], []
        absolute_set_sizes = (relative_set_sizes * size).astype(np.int_)
        missing_elements = size - np.sum(absolute_set_sizes)
        absolute_set_sizes[np.argsort(absolute_set_sizes)[:missing_elements]] += 1

        for j in range(len(unique_labels)):
            label_idx = windows[j]
            window_idx = label_idx[
                windows_idx[j] : windows_idx[j] + absolute_set_sizes[j]
            ]
            windows_idx[j] += absolute_set_sizes[j]
            sampled_features.append(features[window_idx])
            sampled_labels.append(labels[window_idx])

        sampled_features = np.concatenate(sampled_features, axis=0)
        sampled_labels = np.concatenate(sampled_labels, axis=0)
        sampled_data.append((sampled_features, sampled_labels))

    return tuple(sampled_data)


def apply_preprocessors(
    dataset: Dataset, preprocessor_configs: Dict, seed: List[Seed]
) -> Tuple[Dataset, Dict]:
    """
    Applies a list of preprocessors (specified by `preprocessor_configs`) to a dataset.
    `preprocessor_configs` is a dictionary containing the name of the preprocessor as
    key and the configuration as value. The configuration is passed to the preprocessor
    generator function obtained from the `PreprocessorRegistry`.

    Args:
        dataset: Dataset to preprocess.
        preprocessor_configs: A dictionary containing the configurations of the
            preprocessors.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        A tuple containing the preprocessed dataset and a dictionary containing
    """

    preprocess_info = {}
    for idx, (preprocessor_name, preprocessor_config) in enumerate(
        preprocessor_configs.items()
    ):
        preprocessor_fn = PreprocessorRegistry[preprocessor_name]
        dataset, info = preprocessor_fn(dataset, **preprocessor_config, seed=seed[idx])
        preprocess_info.update(
            {f"preprocessor.{preprocessor_name}.{k}": v for k, v in info.items()}
        )

    return dataset, preprocess_info


def flip_labels(
    dataset: Dataset, perc: float = 0.2, seed: Seed = None
) -> Tuple[Dataset, Dict]:
    """
    Flips a percentage of labels in a dataset. The labels are flipped randomly. The
    number of flipped labels is returned in the `preprocess_info` dictionary.

    Args:
        dataset: Dataset to flip labels.
        perc: Number of labels to flip in percent. Must be in the range [0, 1].
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        A tuple containing the dataset with flipped labels and a dictionary containing
        the number and indices of the flipped labels.

    """
    labels = dataset.y_train
    rng = np.random.default_rng(seed)
    num_data_indices = int(perc * len(labels))
    p = rng.permutation(len(labels))[:num_data_indices]
    labels[p] = 1 - labels[p]
    dataset.y_train = labels
    return dataset, {"idx": [int(i) for i in p], "n_flipped": num_data_indices}


PreprocessorRegistry = {"flip_labels": flip_labels}


if __name__ == "__main__":
    sample_data()
