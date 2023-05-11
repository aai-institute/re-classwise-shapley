import json
import os
import pickle
from typing import Dict, Tuple

from pydvl.utils import Dataset

from csshapley22.data.config import Config
from csshapley22.data.preprocess import PreprocessorRegistry
from csshapley22.data.utils import make_hash_sha256
from csshapley22.dataset import create_openml_dataset
from csshapley22.log import setup_logger
from csshapley22.utils import set_random_seed

logger = setup_logger()


def fetch_dataset(dataset_name: str, dataset_kwargs: Dict):
    dataset_idx = make_hash_sha256(dataset_kwargs)
    dataset_folder = Config.RAW_PATH / dataset_idx
    validation_set_path = str(dataset_folder / "validation_set.pkl")
    test_set_path = str(dataset_folder / "test_set.pkl")

    os.makedirs(str(dataset_folder), exist_ok=True)
    logger.info(f"Dataset {dataset_name} doesn't exist.")
    logger.debug(f"Dataset config is \n{dataset_kwargs}.")
    set_random_seed(dataset_kwargs.get("seed", 42))

    dataset_kwargs.pop("preprocessor", None)
    validation_set, test_set = create_openml_dataset(**dataset_kwargs)

    with open(str(dataset_folder / "dataset_kwargs.json"), "w") as file:
        json.dump(dataset_kwargs, file, sort_keys=True, indent=4)

    for set_path, set in [
        (validation_set_path, validation_set),
        (test_set_path, test_set),
    ]:
        with open(set_path, "wb") as file:
            pickle.dump(set, file)

    logger.info(f"Stored dataset '{dataset_name}' on disk.")
