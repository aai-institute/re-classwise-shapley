import json
import os
from typing import Dict

import click
import numpy as np
from dvc.api import params_show

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.config import Config
from csshapley22.data.preprocess import FilterRegistry, PreprocessorRegistry
from csshapley22.log import setup_logger
from csshapley22.utils import set_random_seed

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

    filters = dataset_kwargs.get("filters", None)
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

    os.makedirs(preprocessed_folder, exist_ok=True)
    np.save(preprocessed_folder / "x.npy", x)
    np.save(preprocessed_folder / "y.npy", y)
    logger.info(
        f"Stored preprocessed dataset '{dataset_name}' on disk in folder '{preprocessed_folder}."
    )

    with open(str(preprocessed_folder / "info.json"), "w") as file:
        json.dump(
            info,
            file,
            sort_keys=True,
            indent=4,
        )

    with open(str(preprocessed_folder / "filters.json"), "w") as file:
        json.dump(
            filters,
            file,
            sort_keys=True,
            indent=4,
        )

    with open(str(preprocessed_folder / "preprocess.json"), "w") as file:
        json.dump(
            preprocessor_definitions,
            file,
            sort_keys=True,
            indent=4,
        )


if __name__ == "__main__":
    preprocess_data()
