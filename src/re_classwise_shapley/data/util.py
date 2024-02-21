import glob
import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset, Seed

logger = setup_logger(__name__)


def stratified_sampling(
    features: NDArray[np.float_],
    labels: NDArray[np.int_],
    sizes: Tuple[int],
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


def store_dataset(dataset: RawDataset, output_folder: Path):
    """
    Stores a dataset on disk.
    :param dataset: Tuple of x, y and additional info.
    :param output_folder: Path to the folder where the dataset should be stored.
    """

    try:
        x, y, addition_info = dataset
        logger.info(f"Storing dataset in folder '{output_folder}'.")
        os.makedirs(str(output_folder))
        np.save(str(output_folder / "x.npy"), x)
        np.save(str(output_folder / "y.npy"), y)

        for file_name, content in addition_info.items():
            with open(str(output_folder / file_name), "w") as file:
                json.dump(
                    content,
                    file,
                    sort_keys=True,
                    indent=4,
                )

    except KeyboardInterrupt as e:
        logger.info(f"Removing folder '{output_folder}' due to keyboard interrupt.")
        shutil.rmtree(str(output_folder), ignore_errors=True)
        raise e


def load_dataset(input_folder: Path) -> RawDataset:
    """
    Loads a dataset from disk.
    :param input_folder: Path to the folder containing the dataset.
    :return: Tuple of x, y and additional info.
    """
    x = np.load(str(input_folder / "x.npy"))
    y = np.load(str(input_folder / "y.npy"), allow_pickle=True)

    additional_info = {}
    for file_path in glob.glob(str(input_folder) + "/*.json"):
        with open(file_path, "r") as file:
            file_name = os.path.basename(file_path)
            additional_info[file_name] = json.load(file)

    return x, y, additional_info
