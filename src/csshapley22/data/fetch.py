import json
import os
import pickle
from typing import Dict, Tuple

import click
from dvc.api import params_show
from pydvl.utils import Dataset

from csshapley22.data.config import Config
from csshapley22.data.preprocess import FilterRegistry, PreprocessorRegistry
from csshapley22.data.utils import make_hash_sha256
from csshapley22.dataset import create_openml_dataset
from csshapley22.utils import set_random_seed, setup_logger

logger = setup_logger()


@click.command()
def __fetch_all_datasets():
    fetch_datasets()


def fetch_datasets():
    logger.info("Starting data valuation experiment")
    settings = params_show()["settings"]
    dataset_settings = settings["datasets"]

    logger.info("Fetching datasets.")
    collected_datasets = {}
    for dataset_name, dataset_kwargs in dataset_settings.items():
        validation_set, test_set = fetch_dataset(dataset_name, dataset_kwargs)
        collected_datasets[dataset_name] = (validation_set, test_set)

    return collected_datasets


def fetch_dataset(dataset_name: str, dataset_kwargs: Dict) -> Tuple[Dataset, Dataset]:
    logger.info(f"Fetch dataset '{dataset_name}' into {Config.DATASET_PATH}.")
    dataset_idx = make_hash_sha256(dataset_kwargs)
    dataset_folder = Config.DATASET_PATH / dataset_idx
    validation_set_path = str(dataset_folder / "validation_set.pkl")
    test_set_path = str(dataset_folder / "test_set.pkl")

    if not dataset_folder.exists():
        logger.info(
            f"Dataset {dataset_name} with config {dataset_kwargs} doesn't exist."
        )
        set_random_seed(dataset_kwargs.get("seed", 42))

        preprocessor_definitions = dataset_kwargs.pop("preprocessor", None)
        validation_set, test_set = create_openml_dataset(**dataset_kwargs)

        if preprocessor_definitions is not None:
            for (
                preprocessor_name,
                preprocessor_kwargs,
            ) in preprocessor_definitions.items():
                preprocessor = PreprocessorRegistry[preprocessor_name]
                validation_set, test_set = preprocessor(
                    validation_set, test_set, **preprocessor_kwargs
                )

        os.makedirs(str(dataset_folder))

        with open(str(dataset_folder / "dataset_kwargs.json"), "w") as file:
            json.dump(dataset_kwargs, file, sort_keys=True, indent=4)

        for set_path, set in [
            (validation_set_path, validation_set),
            (test_set_path, test_set),
        ]:
            with open(set_path, "wb") as file:
                pickle.dump(set, file)

        logger.info("Stored datasets on disk.")

    else:
        logger.info(
            f"Dataset {dataset_name} with config {dataset_kwargs} already exists. Skip creation."
        )

        with open(validation_set_path, "rb") as file:
            validation_set = pickle.load(file)

        with open(test_set_path, "rb") as file:
            test_set = pickle.load(file)

        logger.info("Loaded datasets from disk.")

    return validation_set, test_set


if __name__ == "__main__":
    __fetch_all_datasets()
