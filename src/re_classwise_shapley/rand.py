from typing import Tuple

import numpy as np
from numpy._typing import NDArray
from pydvl.utils import Dataset

from re_classwise_shapley.types import Seed

__all__ = ["sample_val_test_set", "stratified_sampling"]


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
