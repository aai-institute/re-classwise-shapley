import os

import click
from dvc.api import params_show
from sklearn.datasets import fetch_openml

from re_classwise_shapley.config import Config
from re_classwise_shapley.io import store_dataset
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset

logger = setup_logger()


@click.command()
@click.option("--dataset-name", type=str, required=True)
def fetch_data(dataset_name: str):
    """
    Fetches a single dataset from openml and stores it on disk. The openml id is taken
    from the `params.datasets.openml_id` section.

    Args:
        dataset_name: The name of the dataset to fetch.
    """
    params = params_show()
    dataset_config = params["datasets"][dataset_name]
    open_ml_id = dataset_config["openml_id"]

    dataset_folder = Config.RAW_PATH / dataset_name
    if os.path.exists(dataset_folder):
        logger.info(f"Dataset {dataset_name} exists. Skipping.")
        return

    logger.info(f"Download dataset {dataset_name} with openml_id {open_ml_id}.")
    dataset = fetch_single_dataset(open_ml_id)
    store_dataset(dataset, dataset_folder)


def fetch_single_dataset(
    openml_id: int,
) -> RawDataset:
    """
    Fetches a single dataset from openml.

    Args:
        openml_id: Openml id of the dataset.
    Returns:
        Tuple of x, y and additional info. Additional information contains a mapping
        from file_names to dictionaries (to be saved as `*.json`). It contains a file
        name `info.json` with information `feature_names`, `target_names` and
        `description`.
    """
    logger.info(f"Downloading dataset with id '{openml_id}'.")
    data = fetch_openml(data_id=openml_id)
    x = data.data.to_numpy().astype(float)
    y = data.target.to_numpy()
    info = {
        "feature_names": data.get("feature_names"),
        "target_names": data.get("target_names"),
        "description": data.get("DESCR"),
    }
    return x, y, {"info.json": info}


if __name__ == "__main__":
    fetch_data()
