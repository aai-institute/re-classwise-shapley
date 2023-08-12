import os
import time
from functools import partial
from typing import Dict

import numpy as np
from pydvl.utils import Dataset

from csshapley22.data.config import Config
from csshapley22.dataset import subsample
from csshapley22.log import setup_logger
from csshapley22.types import (
    ModelGeneratorFactory,
    ValTestSetFactory,
    ValuationMethodsFactory,
)
from csshapley22.utils import instantiate_model
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


def parse_valuation_methods_config(
    valuation_methods: Dict[str, Dict], global_kwargs: Dict = None
) -> ValuationMethodsFactory:
    if global_kwargs is None:
        global_kwargs = {}

    res = {}
    logger.info("Parsing valuation methods...")
    for name, kwargs in valuation_methods.items():
        valuation_method = kwargs.pop("valuation_method")
        res[name] = partial(
            compute_values, valuation_method=valuation_method, **kwargs, **global_kwargs
        )

    return res


def parse_datasets_config(dataset_settings: Dict[str, Dict]) -> ValTestSetFactory:
    logger.info("Parsing datasets...")
    collected_datasets = {}
    for dataset_name, dataset_kwargs in dataset_settings.items():
        collected_datasets[dataset_name] = partial(
            load_single_dataset, dataset_name, dataset_kwargs
        )

    return collected_datasets


def load_single_dataset(dataset_name: str, dataset_kwargs: Dict):
    stratified = dataset_kwargs.get("stratified", True)
    val_size = dataset_kwargs["dev_size"]
    train_size = dataset_kwargs["train_size"]
    test_size = dataset_kwargs["test_size"]

    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_name

    x = np.load(preprocessed_folder / "x.npy")
    y = np.load(preprocessed_folder / "y.npy", allow_pickle=True)

    validation_set, test_set = _encode_and_pack_into_datasets(
        x, y, train_size, val_size, test_size, stratified
    )
    return validation_set, test_set


def parse_models_config(models_config: Dict[str, Dict]) -> ModelGeneratorFactory:
    logger.info("Parsing models...")
    collected_generators = {}
    for model_name, model_kwargs in models_config.items():
        collected_generators[model_name] = partial(
            instantiate_model, model_name=model_name, **model_kwargs
        )

    return collected_generators


def _encode_and_pack_into_datasets(x, y, train_size, val_size, test_size, stratified):
    seed = int(os.getpid() + time.time()) % (2**31 - 1)
    (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = subsample(
        x, y, train_size, val_size, test_size, stratified=stratified, seed=seed
    )
    perm_train = np.random.permutation(len(x_train))
    perm_dev = np.random.permutation(len(x_dev))
    perm_test = np.random.permutation(len(x_test))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]
    validation_set = Dataset(
        x_train,
        y_train,
        x_dev[perm_dev],
        y_dev[perm_dev],
    )
    test_set = Dataset(
        x_train,
        y_train,
        x_test[perm_test],
        y_test[perm_test],
    )
    return validation_set, test_set
