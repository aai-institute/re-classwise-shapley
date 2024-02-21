import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset

logger = setup_logger(__name__)


def store_dataset(dataset: RawDataset, output_folder: Path):
    """
    Stores a dataset on disk. The dataset is stored as `x.npy` and `y.npy`. Additional
    information is stored as `*.json` files.

    Args:
        dataset: Tuple of x, y and additional info.
        output_folder: Path to the folder where the dataset should be stored.
    """

    try:
        x, y, addition_info = dataset
        logger.info(f"Storing dataset in folder '{output_folder}'.")
        os.makedirs(str(output_folder))
        np.save(str(output_folder / "x.npy"), x)
        np.save(str(output_folder / "y.npy"), y)

        for file_name, content in addition_info.items():
            with open(str(output_folder / file_name), "w") as file:
                json.dump(
                    content,
                    file,
                    sort_keys=True,
                    indent=4,
                )

    except KeyboardInterrupt as e:
        logger.info(f"Removing folder '{output_folder}' due to keyboard interrupt.")
        shutil.rmtree(str(output_folder), ignore_errors=True)
        raise e


def load_dataset(input_folder: Path) -> RawDataset:
    """
    Loads a dataset from disk.

    Args:
        input_folder: Path to the folder containing the dataset.
        Tuple of x, y and additional info.
    """
    x = np.load(str(input_folder / "x.npy"))
    y = np.load(str(input_folder / "y.npy"), allow_pickle=True)

    additional_info = {}
    for file_path in glob.glob(str(input_folder) + "/*.json"):
        with open(file_path, "r") as file:
            file_name = os.path.basename(file_path)
            additional_info[file_name] = json.load(file)

    return x, y, additional_info
