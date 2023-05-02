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


def fetch_dataset(dataset_name: str, dataset_kwargs: Dict) -> Tuple[Dataset, Dataset]:
    dataset_idx = make_hash_sha256(dataset_kwargs)
    dataset_folder = Config.DATASET_PATH / dataset_idx
    validation_set_path = str(dataset_folder / "validation_set.pkl")
    test_set_path = str(dataset_folder / "test_set.pkl")

    if not dataset_folder.exists():
        logger.info(f"Dataset {dataset_name} doesn't exist.")
        logger.debug(f"Dataset config is \n{dataset_kwargs}.")
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

        logger.info(f"Stored dataset '{dataset_name}' on disk.")

    else:
        with open(validation_set_path, "rb") as file:
            validation_set = pickle.load(file)

        with open(test_set_path, "rb") as file:
            test_set = pickle.load(file)

        logger.info(f"Loaded dataset '{dataset_name}' from disk.")
        logger.debug(f"Dataset config is \n{dataset_kwargs}.")

    return validation_set, test_set
