import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydvl.utils import Dataset

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset
from re_classwise_shapley.utils import load_params_fast

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


def save_df_as_table(df: pd.DataFrame, path: Union[str, Path]):
    """
    Store a dataframe as an image. It generates a heatmap of the dataframe. This heatmap
    is a table representation of the dataframe.

    Args:
        df: A pd.DataFrame to visualize as a table.
        path: A path to store the file in.
    """
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, cmap=matplotlib.cm.get_cmap("viridis_r"), ax=ax)
    plt.xlabel(df.columns.name)
    plt.ylabel(df.index.name)
    plt.tight_layout()
    fig.savefig(path, transparent=True)


def save_fig_and_log_artifact(
    fig: plt.Figure,
    output_folder: Path,
    file_name: str,
    folder_name: Optional[str] = None,
):
    if folder_name is not None:
        output_folder = output_folder / folder_name
    os.makedirs(output_folder, exist_ok=True)
    output_file = output_folder / file_name
    fig.savefig(output_file, transparent=True)
    mlflow.log_artifact(str(output_file), folder_name)


def dataset_to_dataframe(dataset: Dataset) -> pd.DataFrame:
    x = np.concatenate((dataset.x_train, dataset.x_test), axis=0)
    y = np.concatenate((dataset.y_train, dataset.y_test), axis=0)
    df = pd.DataFrame(np.concatenate((x, y.reshape([-1, 1])), axis=1))
    df.columns = dataset.feature_names + dataset.target_names
    return df


def log_datasets(datasets):
    for _, row in datasets.iterrows():
        for dataset_type in ["val_set", "test_set"]:
            dataset = row[dataset_type]
            dataset_name = row["dataset_name"]
            repetition_id = str(row["repetition_id"])
            mlflow.log_input(
                mlflow.data.pandas_dataset.from_pandas(
                    dataset_to_dataframe(dataset),
                    targets=dataset.target_names[0],
                    name=f"{dataset_name}_{repetition_id}_{dataset_type}",
                ),
                tags={
                    "set": dataset_type,
                    "dataset": dataset_name,
                    "repetition": repetition_id,
                },
            )
