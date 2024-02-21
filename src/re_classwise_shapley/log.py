import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydvl.utils import Dataset


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with a uniform configuration.

    Args:
        name: Optional name of the logger. If no name is given, the name of the
            current module is used.

    Returns:
        A logger with the given name and hard coded configuration.
    """
    logger = logging.getLogger(name or __name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.level = logging.INFO
    return logger


def log_datasets(datasets: pd.DataFrame):
    """
    Logs datasets to MLflow.

    Args:
        datasets (pd.DataFrame): The DataFrame containing the datasets to be logged.
            Required columns:
                - 'val_set' (pydvl.utils.Dataset): The validation dataset.
                - 'test_set' (pydvl.utils.Dataset): The test dataset.
                - 'dataset_name' (str): The name of the dataset.
                - 'repetition_id' (int): The repetition ID.
    """
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


def log_figure(
    fig: plt.Figure,
    output_folder: Path,
    file_name: str,
    namespace: Optional[str] = None,
    store_in_mlflow: bool = True,
):
    """
    Takes a matplotlib Figure and stores it inside an output folder.

    Args:
        fig: The matplotlib figure to be logged.
        output_folder: The path to the output folder where the figure will be stored.
        file_name: The name of the file to save the figure as.
        namespace: Namespace is used as a subfolder to be stored. Default is None.
        store_in_mlflow: Whether to store the figure in MLflow. Default is True.
    """
    output_folder = output_folder / namespace if namespace else output_folder
    os.makedirs(output_folder, exist_ok=True)
    output_file = output_folder / file_name
    fig.savefig(output_file, transparent=True)
    if store_in_mlflow:
        mlflow.log_artifact(str(output_file), namespace)


def dataset_to_dataframe(dataset: Dataset) -> pd.DataFrame:
    """
    Converts a dataset object to a pandas DataFrame.

    Args:
        dataset: The dataset object to be converted.

    Returns:
        The converted dataset as a pandas DataFrame.
    """
    x = np.concatenate((dataset.x_train, dataset.x_test), axis=0)
    y = np.concatenate((dataset.y_train, dataset.y_test), axis=0)
    df = pd.DataFrame(
        np.concatenate((x, y.reshape([-1, 1])), axis=1),
        columns=dataset.feature_names + dataset.target_names,
    )
    return df
