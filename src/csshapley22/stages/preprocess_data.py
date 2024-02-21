import os
import pickle
from typing import Dict

import click
from dvc.api import params_show

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.config import Config
from csshapley22.data.preprocess import PreprocessorRegistry
from csshapley22.data.utils import make_hash_sha256
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
    dataset_idx = make_hash_sha256(dataset_kwargs)
    raw_folder = Config.RAW_PATH / dataset_idx
    validation_set_path = str(raw_folder / "validation_set.pkl")
    test_set_path = str(raw_folder / "test_set.pkl")

    with open(validation_set_path, "rb") as file:
        validation_set = pickle.load(file)

    with open(test_set_path, "rb") as file:
        test_set = pickle.load(file)

    preprocessor_definitions = dataset_kwargs.pop("preprocessor", None)

    if preprocessor_definitions is not None:
        for (
            preprocessor_name,
            preprocessor_kwargs,
        ) in preprocessor_definitions.items():
            preprocessor = PreprocessorRegistry[preprocessor_name]
            validation_set, test_set = preprocessor(
                validation_set, test_set, **preprocessor_kwargs
            )

    preprocessed_folder = Config.PREPROCESSED_PATH / dataset_idx
    os.makedirs(preprocessed_folder, exist_ok=True)
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
