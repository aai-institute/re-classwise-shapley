import json
import os
import pickle
from typing import Dict

import click
import numpy as np
from dvc.api import params_show
from pydvl.utils import Dataset
from sklearn import preprocessing

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.config import Config
from csshapley22.data.preprocess import FilterRegistry, PreprocessorRegistry
from csshapley22.dataset import subsample
from csshapley22.log import setup_logger
from csshapley22.utils import order_dict, set_random_seed

logger = setup_logger()
set_random_seed(RANDOM_SEED)


@click.command()
@click.option("--dataset-name", type=str, required=True)
def preprocess_data(dataset_name: str):
    logger.info(f"Start preprocessing of '{dataset_name}'.")
    params = params_show()
    datasets_settings = params["datasets"]
    dataset_kwargs = datasets_settings[dataset_name]
    preprocess_dataset(dataset_name, dataset_kwargs)
    logger.info(f"Preprocessed '{dataset_name}' with configuration \n{dataset_kwargs}.")


def preprocess_dataset(dataset_name: str, dataset_kwargs: Dict):
    dataset_folder = Config.RAW_PATH / dataset_name
    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_name

    x = np.load(dataset_folder / "x.npy")
    y = np.load(dataset_folder / "y.npy", allow_pickle=True)

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
            x = preprocessor(x, **preprocessor_kwargs)

    stratified = dataset_kwargs.get("stratified", False)
    val_size = dataset_kwargs["dev_size"]
    train_size = dataset_kwargs["train_size"]
    test_size = dataset_kwargs["test_size"]

    test_set, validation_set = _encode_and_pack_into_datasets(
        x, y, info, train_size, val_size, test_size, stratified
    )

    _store_data(
        validation_set, test_set, dataset_name, dataset_kwargs, preprocessed_folder
    )


def _encode_and_pack_into_datasets(
    x, y, info, train_size, val_size, test_size, stratified
):
    (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = subsample(
        x, y, train_size, val_size, test_size, stratified=stratified
    )
    le = preprocessing.LabelEncoder()
    le.fit(y)
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
    )
    test_set = Dataset(
        x_train,
        y_train,
        x_test[perm_test],
        y_test[perm_test],
    )
    return test_set, validation_set


def _store_data(
    validation_set, test_set, dataset_name, dataset_kwargs, preprocessed_folder
):
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


if __name__ == "__main__":
    preprocess_data()
