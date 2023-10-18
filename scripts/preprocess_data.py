"""
Stage 2 for preprocessing datasets fetched in stage 1.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Preprocesses the datasets as defined in the `datasets` section of `params.yaml` file.
All files are stored in `Accessor.PREPROCESSED_PATH / dataset_name` as`x.npy` and
`y.npy`. Additional information is stored in `*.json` files.
"""

import os

import click

from re_classwise_shapley.io import Accessor, load_raw_dataset, store_raw_dataset
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.preprocess import preprocess_dataset
from re_classwise_shapley.utils import load_params_fast

logger = setup_logger("preprocess_data")


@click.command()
@click.option("--dataset-name", type=str, required=True)
def preprocess_data(
    dataset_name: str,
):
    """
    Preprocesses a dataset and stores it on disk. The preprocessing steps are defined in
    the `params.datasets` section. The dataset is stored as `x.npy` and `y.npy`.
    Additional information is stored as `*.json` files. All of them are stored in a
    folder `Access.PREPROCESSED_PATH / dataset_name`.

    Args:
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
    """
    _preprocess_data(dataset_name)


def _preprocess_data(
    dataset_name: str,
):
    preprocessed_folder = Accessor.PREPROCESSED_PATH / dataset_name
    if os.path.exists(preprocessed_folder / "x.npy") and os.path.exists(
        preprocessed_folder / "y.npy"
    ):
        return logger.info(
            f"Preprocessed data exists in '{preprocessed_folder}'. Skipping..."
        )

    params = load_params_fast()
    datasets_settings = params["datasets"]

    dataset_folder = Accessor.RAW_PATH / dataset_name
    logger.info(f"Loading raw dataset '{dataset_name}' from {dataset_folder}.")
    raw_dataset = load_raw_dataset(dataset_folder)

    logger.info(f"Preprocessing dataset '{dataset_name}'.")
    dataset_kwargs = datasets_settings[dataset_name]
    preprocessed_dataset = preprocess_dataset(raw_dataset, dataset_kwargs)
    store_raw_dataset(preprocessed_dataset, preprocessed_folder)


if __name__ == "__main__":
    preprocess_data()
