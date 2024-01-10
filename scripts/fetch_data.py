"""
Stage 1 for fetching the data from openml.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Downloads the datasets from openml as defined in the `params.yaml` file. All files are
stored in `Accessor.RAW_PATH / dataset_name` as`x.npy` and `y.npy`. Additional
information is stored in `*.json` files.
"""

import click

from re_classwise_shapley.io import (
    Accessor,
    fetch_openml_raw_dataset,
    has_raw_dataset,
    store_raw_dataset,
)
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("fetch_data")


@click.command()
@click.option("--dataset-name", type=str, required=True)
def fetch_data(dataset_name: str):
    """
    Fetches a single dataset from openml and stores it on disk. The openml id is taken
    from the `params.datasets.openml_id` section. The dataset is stored as `x.npy` and
    `y.npy`. Additional information is stored as `*.json` files. All of them are
    stored in a folder `Access.RAW_PATH / dataset_name`.

    Args:
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
    """
    _fetch_data(dataset_name)


def _fetch_data(dataset_name: str):
    params = load_params_fast()
    dataset_config = params["datasets"][dataset_name]
    open_ml_id = dataset_config["openml_id"]

    dataset_folder = Accessor.RAW_PATH / dataset_name
    if has_raw_dataset(dataset_folder):
        return logger.info(f"Dataset {dataset_name} exists. Skipping...")

    logger.info(f"Download dataset {dataset_name} with openml_id {open_ml_id}.")
    dataset = fetch_openml_raw_dataset(open_ml_id)
    store_raw_dataset(dataset, dataset_folder)


if __name__ == "__main__":
    fetch_data()
