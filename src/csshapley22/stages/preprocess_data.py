import json
import os
import pickle
from typing import Dict, Union

import click
import numpy as np
from dvc.api import params_show
from numpy._typing import NDArray
from pydvl.utils import Dataset
from sklearn import preprocessing

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.config import Config
from csshapley22.data.preprocess import FilterRegistry, PreprocessorRegistry
from csshapley22.data.utils import make_hash_sha256
from csshapley22.dataset import subsample
from csshapley22.log import setup_logger
from csshapley22.utils import set_random_seed

logger = setup_logger()
set_random_seed(RANDOM_SEED)


@click.command()
def preprocess_data():
    logger.info("Starting downloading of data.")

    params = params_show()
    general_settings = params["general"]

    # fetch datasets
    datasets_settings = general_settings["datasets"]
    for dataset_name, dataset_kwargs in datasets_settings.items():
        logger.info(f"Fetching dataset {dataset_name} with kwargs {dataset_kwargs}.")
        preprocess_dataset(dataset_name, dataset_kwargs)


def preprocess_dataset(dataset_name: str, dataset_kwargs: Dict):
    openml_id = dataset_kwargs["openml_id"]
    dataset_folder = Config.RAW_PATH / str(openml_id)

    x = np.load(dataset_folder / "x.npy")
    y = np.load(dataset_folder / "y.npy")

    with open(str(dataset_folder / "info.json"), "r") as file:
        info = json.load(file)

    filters = dataset_kwargs.get("filter", None)
    if filters is not None:
        for filter_name, filter_kwargs in filters.items():
            data_filter = FilterRegistry[filter_name]
            x, y = data_filter(x, y, **filter_kwargs)

    preprocessor_definitions = dataset_kwargs.pop("preprocessor", None)
    if preprocessor_definitions is not None:
        for (
            preprocessor_name,
            preprocessor_kwargs,
        ) in preprocessor_definitions.items():
            preprocessor = PreprocessorRegistry[preprocessor_name]
            x, y = preprocessor(x, y, **preprocessor_kwargs)

    stratified = dataset_kwargs.get("stratified", False)
    val_size = dataset_kwargs["dev_size"]
    train_size = dataset_kwargs["train_size"]
    test_size = dataset_kwargs["test_size"]

    test_set, validation_set = _encode_and_pack_into_datasets(
        x, y, info, train_size, val_size, test_size, stratified
    )

    _store_data(validation_set, test_set, dataset_name, dataset_kwargs)


def _encode_and_pack_into_datasets(
    x, y, info, train_size, val_size, test_size, stratified
):
    (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = subsample(
        x, y, train_size, val_size, test_size, stratified=stratified
    )
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((y_train, y_test, y_dev)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_dev = le.transform(y_dev)
    perm_train = np.random.permutation(len(x_train))
    perm_dev = np.random.permutation(len(x_dev))
    perm_test = np.random.permutation(len(x_test))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]
    validation_set = Dataset(
        x_train,
        y_train[perm_train],
        x_dev[perm_dev],
        y_dev[perm_dev],
        **info,
    )
    test_set = Dataset(
        x_train,
        y_train,
        x_test[perm_test],
        y_test[perm_test],
        **info,
    )
    return test_set, validation_set


def _store_data(validation_set, test_set, dataset_name, dataset_kwargs):
    dataset_idx = make_hash_sha256(dataset_kwargs)
    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_idx
    os.makedirs(preprocessed_folder, exist_ok=True)
    with open(str(preprocessed_folder / "dataset_kwargs.json"), "w") as file:
        json.dump(dataset_kwargs, file, sort_keys=True, indent=4)
    validation_set_path = str(preprocessed_folder / "validation_set.pkl")
    test_set_path = str(preprocessed_folder / "test_set.pkl")
    for set_path, set in [
        (validation_set_path, validation_set),
        (test_set_path, test_set),
    ]:
        with open(set_path, "wb") as file:
            pickle.dump(set, file)
    logger.info(f"Stored dataset '{dataset_name}' on disk.")


def process(x: NDArray[np.float_], y: NDArray[Union[np.float_, np.int_]]):
    return val_dataset, test_dataset


if __name__ == "__main__":
    preprocess_data()
