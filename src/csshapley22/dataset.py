from enum import Enum
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np
import openml
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


def create_openml_dataset(
    openml_id: int,
    train_size: int,
    dev_size: int,
    test_size: int,
    filters: Dict = None,
) -> Tuple[Dataset, Dataset]:
    data = fetch_openml(data_id=openml_id)
    X = data.data.to_numpy()
    y = data.target.to_numpy()

    if filters is not None:
        for filter_name, filter_kwargs in filters.items():
            data_filter = FilterRegistry[filter_name]
            X, y = data_filter(X, y, **filter_kwargs)

    # sample some subsets for the datasets
    num_data = train_size + dev_size + test_size
    p = np.random.permutation(len(X))[:num_data]
    train_idx = p[:train_size]
    dev_idx = p[train_size : train_size + dev_size]
    test_idx = p[train_size + dev_size :]

    x_train = X[train_idx]
    y_train = y[train_idx]
    x_dev = X[dev_idx]
    y_dev = y[dev_idx]
    x_test = X[test_idx]
    y_test = y[test_idx]

    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((y_train, y_test, y_dev)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_dev = le.transform(y_dev)

    val_dataset = Dataset(
        x_train,
        y_train,
        x_dev,
        y_dev,
        feature_names=data.get("feature_names"),
        target_names=data.get("target_names"),
        description=data.get("DESCR"),
    )
    test_dataset = Dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names=data.get("feature_names"),
        target_names=data.get("target_names"),
        description=data.get("DESCR"),
    )

    return val_dataset, test_dataset


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
