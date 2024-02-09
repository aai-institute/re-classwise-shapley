import glob
import json
import os
import pickle
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from pydvl.utils import Dataset
from sklearn.datasets import fetch_openml

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import OneOrMany, RawDataset, ensure_list

__all__ = [
    "store_raw_dataset",
    "load_raw_dataset",
    "has_val_test_dataset",
    "has_raw_dataset",
    "fetch_openml_raw_dataset",
    "Accessor",
]

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
    logger.info(f"Loading raw dataset from {input_folder}.")
    x = np.load(str(input_folder / "x.npy"))
    y = np.load(str(input_folder / "y.npy"), allow_pickle=True)

    additional_info = {}
    for file_path in glob.glob(str(input_folder) + "/*.json"):
        with open(file_path, "r") as file:
            file_name = os.path.basename(file_path)
            additional_info[file_name] = json.load(file)

    return x, y, additional_info


def has_raw_dataset(dataset_folder: Path) -> bool:
    """
    Checks if the dataset files are present in the given dataset folder.

    The function verifies the existence of 'x.npy' and 'y.npy' files in the
    provided folder path. Both files are required for dataset.

    Args:
        dataset_folder: The path of the folder where dataset files are supposed to exist.

    Returns:
        True if both 'x.npy' and 'y.npy' files exist, False otherwise.
    """
    return os.path.exists(dataset_folder / "x.npy") and os.path.exists(
        dataset_folder / "y.npy"
    )


def fetch_openml_raw_dataset(
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


def has_val_test_dataset(dataset_folder: Path) -> bool:
    """
    Checks if the validation and test dataset files are present in the given dataset
    folder.

    This function verifies the existence of 'val_set.pkl' and 'test_set.pkl' files in
    the provided folder path. Both files are expected for validation and testing
    datasets respectively.

    Args:
        dataset_folder: The path of the folder where validation and test dataset files
            are supposed to exist.

    Returns:
        True if both 'val_set.pkl' and 'test_set.pkl' files exist, False otherwise.
    """
    return os.path.exists(dataset_folder / "val_set.pkl") and os.path.exists(
        dataset_folder / "test_set.pkl"
    )


def store_val_test_data(
    val_set: Dataset,
    test_set: Dataset,
    preprocess_info: Dict[str, Any],
    dataset_folder: Path,
):
    """
    Stores validation and test datasets along with preprocessing information in the
    specified dataset folder.

    The function saves the validation and test datasets as pickle files named
    'val_set.pkl' and 'test_set.pkl' respectively in the given folder path. If the
    preprocessing information is provided and is not empty, it is stored in a JSON file
    named 'preprocess_info.json'.

    Args:
        val_set: The validation dataset to be stored.
        test_set: The test dataset to be stored.
        preprocess_info: A dictionary containing preprocessing information.
        dataset_folder: The path of the folder where the datasets and preprocessing information are to be stored.
    """
    os.makedirs(dataset_folder, exist_ok=True)
    with open(dataset_folder / "val_set.pkl", "wb") as file:
        pickle.dump(val_set, file)
    with open(dataset_folder / "test_set.pkl", "wb") as file:
        pickle.dump(test_set, file)
    if preprocess_info and len(preprocess_info) > 0:
        with open(dataset_folder / "preprocess_info.json", "w") as file:
            json.dump(preprocess_info, file, indent=4, sort_keys=True)


def walker_product_space(raise_if_not_found: bool = True):
    def _fn(
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
            for folder_instance in product(*product_space):
                try:
                    row = fn(*folder_instance)
                    rows.append(row)
                except ValueError as e:
                    if raise_if_not_found:
                        raise e

            return pd.DataFrame(rows)

        return wrapped_walk_product_space

    return _fn


class Accessor:
    """
    Accessor class to load data from the results directory.
    """

    OUTPUT_PATH = Path("./output")
    RAW_PATH = OUTPUT_PATH / "raw"
    PREPROCESSED_PATH = OUTPUT_PATH / "preprocessed"
    THRESHOLD_CHARACTERISTICS_PATH = OUTPUT_PATH / "threshold_characteristics"
    SAMPLED_PATH = OUTPUT_PATH / "sampled"
    VALUES_PATH = OUTPUT_PATH / "values"
    RESULT_PATH = OUTPUT_PATH / "results"
    PLOT_PATH = OUTPUT_PATH / "plots"

    @staticmethod
    @walker_product_space(raise_if_not_found=False)
    def threshold_characteristics_results(
        experiment_name: str,
        dataset_name: str,
        repetition_id: int,
    ) -> Dict:
        folder = (
            Accessor.THRESHOLD_CHARACTERISTICS_PATH
            / experiment_name
            / dataset_name
            / str(repetition_id)
        )
        if not os.path.exists(folder):
            raise ValueError
        curves = pd.read_csv(folder / "threshold_characteristics_curves.csv", sep=";")
        curves = curves.set_index(curves.columns[0])
        stats = pd.read_csv(folder / "threshold_characteristics_stats.csv")
        return {
            "dataset_name": dataset_name,
            "curves": curves,
            "stats": stats,
        }

    @staticmethod
    @walker_product_space()
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
    @walker_product_space()
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
    @walker_product_space()
    def datasets(
        experiment_name: str,
        dataset_name: str,
    ) -> Dict:
        """
        Load datasets from the specified directory.

        Args:
            experiment_name: The name of the experiment.
            dataset_name: The name of the dataset.

        Returns:
            A dictionary containing the loaded datasets and relevant information.
        """
        base_path = Accessor.SAMPLED_PATH / experiment_name / dataset_name
        with open(base_path / "val_set.pkl", "rb") as file:
            val_set = pickle.load(file)

        with open(base_path / "test_set.pkl", "rb") as file:
            test_set = pickle.load(file)

        row = {
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
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
