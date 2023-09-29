import os
from typing import Dict

import click
import pandas as pd
from sklearn import preprocessing

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.filter import FilterRegistry
from re_classwise_shapley.io import load_dataset, store_dataset
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.preprocess import PreprocessorRegistry
from re_classwise_shapley.types import RawDataset
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger()


@click.command()
@click.option("--dataset-name", type=str, required=True)
def preprocess_data(
    dataset_name: str,
):
    """
    Preprocesses a dataset and stores it on disk. The preprocessing steps are defined in
    the `params.datasets` section. The dataset is stored as `x.npy` and `y.npy`.
    Additional information is stored as `*.json` files. All of them are stored in a
    folder `Access.PREPROCESSED_PATH / dataset_name`.

    Args:
        dataset_name: The name of the dataset to preprocess.
    """
    preprocessed_folder = Accessor.PREPROCESSED_PATH / dataset_name
    if os.path.exists(preprocessed_folder):
        logger.info(
            f"Preprocessed data '{dataset_name}' exists in '{preprocessed_folder}'."
        )
        return

    params = load_params_fast()
    datasets_settings = params["datasets"]

    dataset_folder = Accessor.RAW_PATH / dataset_name
    logger.info(f"Loading raw dataset '{dataset_name}' from {dataset_folder}.")
    raw_dataset = load_dataset(dataset_folder)

    logger.info(f"Preprocessing dataset '{dataset_name}'.")
    dataset_kwargs = datasets_settings[dataset_name]
    preprocessed_dataset = preprocess_dataset(raw_dataset, dataset_kwargs)
    store_dataset(preprocessed_dataset, preprocessed_folder)


def preprocess_dataset(raw_dataset: RawDataset, dataset_kwargs: Dict) -> RawDataset:
    """
    Preprocesses a dataset and returns preprocessed data.

    Args:
        raw_dataset: The raw dataset to preprocess.
        dataset_kwargs: The dataset kwargs for processing. Contains the keys `filters`
            and `preprocessor`. The `filters` key contains a dictionary of filters to
            apply. The `preprocessor` key contains a dictionary of preprocessors to
            apply.

    Returns:
        The preprocessed dataset as a tuple of x, y and additional info. Additional
        information contains a mapping from file_names to dictionaries (to be saved as
        `*.json`). It contains a file name `info.json` with information `feature_names`,
        `target_names` and `description`. It also contains a file name `filters.json`
        with the applied filters and a file name `preprocess.json` with the applied
        preprocessors.
    """
    x, y, additional_info = raw_dataset

    filters = dataset_kwargs.get("filters", None)
    if filters is not None:
        for filter_name, filter_kwargs in filters.items():
            logger.info(f"Applying filter '{filter_name}'.")
            data_filter = FilterRegistry[filter_name]
            x, y = data_filter(x, y, **filter_kwargs)

    logger.info(f"Applying preprocessors.")
    preprocessor_definitions = dataset_kwargs.pop("preprocessor", None)
    if preprocessor_definitions is not None:
        for (
            preprocessor_name,
            preprocessor_kwargs,
        ) in preprocessor_definitions.items():
            logger.info(f"Applying preprocessor '{preprocessor_name}'.")
            preprocessor = PreprocessorRegistry[preprocessor_name]
            x, y = preprocessor(x, y, **preprocessor_kwargs)

    logger.info(f"Encoding labels to integers.")
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    additional_info["info.json"]["label_distribution"] = (
        pd.value_counts(y) / len(y)
    ).to_dict()
    additional_info["filters.json"] = filters
    additional_info["preprocess.json"] = preprocessor_definitions
    return x, y, additional_info


if __name__ == "__main__":
    preprocess_data()
