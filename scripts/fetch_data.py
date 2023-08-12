import json
import os

import click
import numpy as np
from dvc.api import params_show
from sklearn.datasets import fetch_openml

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.config import Config
from csshapley22.log import setup_logger
from csshapley22.utils import set_random_seed

logger = setup_logger()
set_random_seed(RANDOM_SEED)


@click.command()
@click.option("--dataset-name", type=str, required=True)
def fetch_data(dataset_name: str):
    logger.info("Starting downloading of data.")
    params = params_show()
    open_ml_id = params["datasets"][dataset_name]["openml_id"]
    logger.info(f"Fetching dataset {dataset_name} with openml_id {open_ml_id}.")
    fetch_single_dataset(dataset_name, open_ml_id)


def fetch_single_dataset(dataset_name: str, openml_id: int):
    dataset_folder = Config.RAW_PATH / dataset_name
    if os.path.exists(dataset_folder):
        logger.info(f"Dataset {openml_id} exist.")
        return

    os.makedirs(str(dataset_folder))
    logger.info(f"Dataset {dataset_name} doesn't exist.")

    data = fetch_openml(data_id=openml_id)
    x = data.data.to_numpy().astype(float)
    y = data.target.to_numpy()

    np.save(dataset_folder / "x.npy", x)
    np.save(dataset_folder / "y.npy", y)
    logger.info(f"Stored dataset '{dataset_name}' on disk in folder '{dataset_folder}.")

    with open(str(dataset_folder / "info.json"), "w") as file:
        json.dump(
            {
                "feature_names": data.get("feature_names"),
                "target_names": data.get("target_names"),
                "description": data.get("DESCR"),
            },
            file,
            sort_keys=True,
            indent=4,
        )


if __name__ == "__main__":
    fetch_data()
