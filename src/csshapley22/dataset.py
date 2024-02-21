from enum import Enum
from functools import partial
from typing import Callable, Dict

import numpy as np
import openml
from numpy.typing import NDArray
from pydvl.utils.dataset import Dataset
from sklearn import preprocessing
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split

__all__ = ["create_diabetes_dataset"]


def flip_labels(
    y: NDArray[np.int_], percentage: float, *, random_state: np.random.RandomState
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = random_state.choice(
        np.arange(len(y)), size=int(percentage * len(y)), replace=False
    )
    y = y.copy()
    y[indices] = np.logical_not(y[indices])
    return y, indices


def create_openml_dataset(train_size: float, dataset_id: int) -> Dataset:
    le = preprocessing.LabelEncoder()
    dataset = Dataset.from_sklearn(
        fetch_openml(data_id=dataset_id), train_size=train_size
    )

    le.fit(np.concatenate((dataset.y_train, dataset.y_test)))
    dataset.y_train = le.transform(dataset.y_train)
    dataset.y_test = le.transform(dataset.y_test)
    return dataset


DatasetRegistry: Dict[str, Callable[[...], Dataset]] = {
    "diabetes": partial(create_openml_dataset, dataset_id=37),
    "click": partial(create_openml_dataset, dataset_id=1216),
}
