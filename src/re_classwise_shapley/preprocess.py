import os
import time
from functools import partial
from typing import Dict, Tuple

import numpy as np
from pydvl.utils import Dataset

from re_classwise_shapley.config import Config
from re_classwise_shapley.data.util import load_dataset, stratified_sampling
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.types import (
    ModelGeneratorFactory,
    Seed,
    ValTestSetFactory,
    ValuationMethodsFactory,
)
from re_classwise_shapley.valuation_methods import compute_values

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


def fetch_and_sample_val_test_dataset(
    dataset_name: str, dataset_kwargs: Dict, seed: Seed = None
) -> Tuple[Dataset, Dataset]:
    val_size = dataset_kwargs["dev_size"]
    train_size = dataset_kwargs["train_size"]
    test_size = dataset_kwargs["test_size"]

    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_name
    x, y, _ = load_dataset(preprocessed_folder)
    validation_set, test_set = _encode_and_pack_into_datasets(
        x, y, (train_size, val_size, test_size), seed=seed
    )
    return validation_set, test_set


def _encode_and_pack_into_datasets(
    x, y, sizes: Tuple[int, int, int], *, seed: Seed = None
):
    (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = stratified_sampling(
        x, y, sizes, seed=seed
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
