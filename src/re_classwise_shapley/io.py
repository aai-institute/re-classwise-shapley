import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


def load_results_per_dataset_and_method(
    experiment_path: Path, metrics: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]]:
    """
    Load the results per dataset and method. The results are loaded from the
    `experiment_path` directory. The results are loaded from the `metric_names` files.

    Args:
        experiment_path: Path to the experiment directory.
        metrics: List of metric names to load.

    Returns:
        A dictionary of dictionaries containing realizations of the distribution over
        curves and metrics.
    """
    params = load_params_fast()
    params_active = params["active"]
    repetitions = params_active["repetitions"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    curves_per_dataset = {}

    for dataset_name in dataset_names:
        dataset_path = experiment_path / dataset_name

        curves_per_valuation_method = {}

        for valuation_method in valuation_methods:
            curves_per_metric = {}
            for repetition in repetitions:
                repetition_path = dataset_path / f"{repetition}" / valuation_method
                for key, metric_config in metrics.items():
                    if key not in curves_per_metric:
                        curves_per_metric[key] = []

                    logger.info(f"Loading metric {key} from path '{repetition_path}'.")
                    metric = pd.read_csv(repetition_path / f"{key}.csv")
                    metric = metric.drop(columns=[metric.columns[0]])
                    metric.index = [key]
                    metric.index.name = key
                    metric.columns = [repetition]

                    curve = pd.read_csv(repetition_path / f"{key}.curve.csv")
                    curve.index = curve[curve.columns[0]]
                    curve = curve.drop(columns=[curve.columns[0]])

                    len_curve_perc = metric_config.get("len_curve_perc", 1)
                    curve = curve.iloc[: int(len_curve_perc * len(curve))]
                    curves_per_metric[key].append((metric, curve))

            curves_per_metric = {
                k: (
                    pd.concat([t[0] for t in v], axis=1),
                    pd.concat([t[1] for t in v], axis=1),
                )
                for k, v in curves_per_metric.items()
            }
            curves_per_valuation_method[valuation_method] = curves_per_metric

        curves_per_dataset[dataset_name] = curves_per_valuation_method

    return curves_per_dataset


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
