from enum import Enum
from functools import partial
from typing import Callable, Dict, Tuple, Union

import numpy as np
import openml
from numpy import ndarray
from numpy.typing import NDArray
from pydvl.utils.dataset import Dataset
from sklearn import preprocessing
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split

from csshapley22.data.preprocess import FilterRegistry

__all__ = ["create_openml_dataset"]


def flip_labels(
    y: NDArray[np.int_], percentage: float, *, random_state: np.random.RandomState
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = random_state.choice(
        np.arange(len(y)), size=int(percentage * len(y)), replace=False
    )
    y = y.copy()
    y[indices] = np.logical_not(y[indices])
    return y, indices


def subsample(
    features: NDArray[np.float_],
    labels: NDArray[np.int_],
    *sizes: int,
    stratified: bool = True
) -> tuple[tuple[NDArray[np.float_], NDArray[np.int_]], ...]:
    """
    Sub-samples a dataset into different sets. It supports normal sampling and
    stratified sampling.

    :param features: Array with the features to be sub-sampled.
    :param labels: Array containing the label data. Same size as ``features``.
    :param sizes: List of the test set sizes which shall be achieved.
    :param stratified: True, if the label distribution shall be reproduced in each
        subset.
    :return: A tuple
    """

    if stratified:
        p = np.random.permutation(len(features))
        features = features[p]
        labels = labels[p]
        unique_labels, num_labels = np.unique(labels, return_counts=True)
        label_indices = [np.argwhere(labels == label)[:, 0] for label in unique_labels]
        relative_set_sizes = num_labels / len(labels)

        data = [list() for _ in sizes]
        it_idx = [0 for _ in unique_labels]

        for i, size in enumerate(sizes):
            absolute_set_sizes = (relative_set_sizes * size).astype(np.int_)
            absolute_set_sizes[-1] = size - np.sum(absolute_set_sizes[:-1])

            for j in range(len(unique_labels)):
                label_idx = label_indices[j]
                window_idx = label_idx[it_idx[j] : it_idx[j] + absolute_set_sizes[j]]
                it_idx[j] += absolute_set_sizes[j]
                data[i].append((features[window_idx], labels[window_idx]))

        data = [
            (np.concatenate([t[0] for t in lst]), np.concatenate([t[1] for t in lst]))
            for lst in data
        ]

    else:
        data = list()
        p = np.random.permutation(len(features))
        it_idx = 0
        for i, size in enumerate(sizes):
            window_idx = p[it_idx : it_idx + size]
            it_idx += size
            data.append((features[window_idx], labels[window_idx]))

    return tuple(data)


def pca_feature_transformer(
    dataset: Tuple[Dataset, Dataset]
) -> Tuple[Dataset, Dataset]:
    pass


def label_binarization(datasets: Tuple[Dataset, Dataset]) -> Dataset:
    pass


def chain_transformer(
    dataset_producer: Callable[[...], Dataset],
    *transformers: Callable[[Tuple[Dataset, Dataset]], Tuple[Dataset, Dataset]]
) -> Callable[[...], Tuple[Dataset, Dataset]]:
    def _transformer(*args, **kwargs):
        dataset = dataset_producer(*args, **kwargs)
        for transformer in transformers:
            dataset = transformer(dataset)

        return dataset

    return _transformer
