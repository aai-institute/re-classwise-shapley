import glob
import json
import os
import pickle
import shutil
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import OneOrMany, RawDataset, ensure_list

__all__ = ["store_raw_dataset", "load_raw_dataset", "Accessor"]

logger = setup_logger(__name__)


def store_raw_dataset(dataset: RawDataset, output_folder: Path):
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


def load_raw_dataset(input_folder: Path) -> RawDataset:
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


def walker_product_space(
    fn: Callable[[Any, ...], Dict]
) -> Callable[[OneOrMany[Any], ...], pd.DataFrame]:
    """
    A decorator that applies a given function to each combination of input instances.

    Args:
        fn: The function to be applied.

    Returns:
        A wrapped function that applies to each combination of input instances `fn`.
    """

    def wrapped_walk_product_space(
        *product_space: OneOrMany[Any],
    ) -> pd.DataFrame:
        """
        Wrapped function that walks through a product space and applies the given
        function.

        Args:
            product_space: The product space to iterate over.

        Returns:
            A DataFrame containing the results of applying the function to
                each combination of input instances.
        """
        product_space = list(map(ensure_list, product_space))
        rows = []
        pbar = tqdm(
            list(product(*product_space)),
            ncols=120,
        )
        for folder_instance in pbar:
            pbar.desc = f"Processing: {folder_instance}"
            row = fn(*folder_instance)
            rows.append(row)

        return pd.DataFrame(rows)

    return wrapped_walk_product_space


class Accessor:
    """
    Accessor class to load data from the results directory.
    """

    OUTPUT_PATH = Path("./output")
    RAW_PATH = OUTPUT_PATH / "raw"
    PREPROCESSED_PATH = OUTPUT_PATH / "preprocessed"
    SAMPLED_PATH = OUTPUT_PATH / "sampled"
    VALUES_PATH = OUTPUT_PATH / "values"
    RESULT_PATH = OUTPUT_PATH / "results"
    PLOT_PATH = OUTPUT_PATH / "plots"

    @staticmethod
    @walker_product_space
    def valuation_results(
        experiment_name: str,
        model_name: str,
        dataset_name: str,
        repetition_id: int,
        method_name: str,
    ) -> Dict:
        """
        Load valuation results from the results directory.

        Args:
            experiment_name: The name of the experiment.
            model_name: The name of the model.
            dataset_name: The name of the dataset.
            repetition_id: The repetition ID.
            method_name: The name of the method.

        Returns:
            A dictionary containing the valuation results and statistics.
        """

        base_path = (
            Accessor.VALUES_PATH
            / experiment_name
            / model_name
            / dataset_name
            / str(repetition_id)
        )
        with open(base_path / f"valuation.{method_name}.pkl", "rb") as f:
            valuation = pickle.load(f)
        with open(base_path / f"valuation.{method_name}.stats.json", "r") as f:
            stats = json.load(f)
        return {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "method_name": method_name,
            "repetition_id": repetition_id,
            "valuation": valuation,
        } | stats

    @staticmethod
    @walker_product_space
    def metrics_and_curves(
        experiment_name: str,
        model_name: str,
        dataset_name: str,
        method_name: str,
        repetition_id: int,
        metric_name: str,
    ) -> Dict:
        """
        Load metrics and curves from the results directory.

        Args:
            experiment_name: The name of the experiment.
            model_name: The name of the model.
            dataset_name: The name of the dataset.
            method_name: The name of the method.
            repetition_id: The repetition ID.
            metric_name: The name of the metric.

        Returns:
            A dictionary containing the metrics and curves.
        """
        base_path = (
            Accessor.RESULT_PATH
            / experiment_name
            / model_name
            / dataset_name
            / str(repetition_id)
            / method_name
        )
        metric = pd.read_csv(base_path / f"{metric_name}.csv")
        metric = metric.iloc[-1, -1]

        curve = pd.read_csv(base_path / f"{metric_name}.curve.csv")
        curve.index = curve[curve.columns[0]]
        curve = curve.drop(columns=[curve.columns[0]]).iloc[:, -1]

        return {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "method_name": method_name,
            "repetition_id": repetition_id,
            "metric_name": metric_name,
            "metric": metric,
            "curve": curve,
        }

    @staticmethod
    @walker_product_space
    def datasets(
        experiment_name: str,
        dataset_name: str,
        repetition_id: int,
    ) -> Dict:
        """
        Load datasets from the specified directory.

        Args:
            experiment_name: The name of the experiment.
            dataset_name: The name of the dataset.
            repetition_id: The repetition ID.

        Returns:
            A dictionary containing the loaded datasets and relevant information.
        """
        base_path = (
            Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
        )
        with open(base_path / f"val_set.pkl", "rb") as file:
            val_set = pickle.load(file)

        with open(base_path / f"test_set.pkl", "rb") as file:
            test_set = pickle.load(file)

        row = {
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "repetition_id": repetition_id,
            "val_set": val_set,
            "test_set": test_set,
        }
        path_preprocess_info = base_path / "preprocess_info.json"
        if os.path.exists(path_preprocess_info):
            with open(path_preprocess_info, "r") as file:
                preprocess_info = json.load(file)
        else:
            preprocess_info = {}

        row["preprocess_info"] = preprocess_info
        return row
