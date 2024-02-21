import os

import click
from sklearn.datasets import fetch_openml

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.io import store_dataset
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger()


@click.command()
@click.option("--dataset-name", type=str, required=True)
def fetch_data(dataset_name: str):
    """
    Fetches a single dataset from openml and stores it on disk. The openml id is taken
    from the `params.datasets.openml_id` section. The dataset is stored as `x.npy` and
    `y.npy`. Additional information is stored as `*.json` files. All of them are
    stored in a folder `Access.RAW_PATH / dataset_name`.

    Args:
        dataset_name: The name of the dataset to fetch.
    """
    params = load_params_fast()
    dataset_config = params["datasets"][dataset_name]
    open_ml_id = dataset_config["openml_id"]

    dataset_folder = Accessor.RAW_PATH / dataset_name
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
