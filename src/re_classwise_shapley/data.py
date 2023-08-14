from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Dataset

from re_classwise_shapley.config import Config
from re_classwise_shapley.io import load_dataset
from re_classwise_shapley.types import Seed


def fetch_and_sample_val_test_dataset(
    dataset_name: str, split_info: Dict, seed: Seed = None
) -> Tuple[Dataset, Dataset]:
    """
    Fetches the dataset and samples the validation and test set.
    :param dataset_name: Name of the dataset.
    :param split_info: Dictionary containing the split information for train, val and
        test set.
    :param seed: Seed to use for permutation and sampling.
    :return: Tuple containing the validation and test set.
    """
    rng = np.random.default_rng(seed)
    val_size = split_info["val_size"]
    train_size = split_info["train_size"]
    test_size = split_info["test_size"]

    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_name
    x, y, _ = load_dataset(preprocessed_folder)
    p = rng.permutation(len(x))
    x, y = x[p], y[p]
    set_train, set_val, set_test = stratified_sampling(
        x, y, (train_size, val_size, test_size), seed=seed.spawn(1)[0]
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
