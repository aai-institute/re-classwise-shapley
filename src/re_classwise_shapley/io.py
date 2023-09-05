import glob
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset, ValuationMethodDict
from re_classwise_shapley.valuation_methods import compute_values

logger = setup_logger(__name__)


def store_dataset(dataset: RawDataset, output_folder: Path):
    """
    Stores a dataset on disk.
    :param dataset: Tuple of x, y and additional info.
    :param output_folder: Path to the folder where the dataset should be stored.
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
    :param input_folder: Path to the folder containing the dataset.
    :return: Tuple of x, y and additional info.
    """
    x = np.load(str(input_folder / "x.npy"))
    y = np.load(str(input_folder / "y.npy"), allow_pickle=True)

    additional_info = {}
    for file_path in glob.glob(str(input_folder) + "/*.json"):
        with open(file_path, "r") as file:
            file_name = os.path.basename(file_path)
            additional_info[file_name] = json.load(file)

    return x, y, additional_info


def parse_valuation_method_dict(
    valuation_method_configs: Dict[str, Dict],
    global_kwargs: Dict = None,
    active_valuation_methods: Optional[List[str]] = None,
    kwargs_key: str = "kwargs",
) -> ValuationMethodDict:
    """
    Parse the valuation method configs to a dictionary mapping strs to function. A
    function accepts a utility and returns a valuation result.

    :param valuation_method_configs: Configuration of all available valuation methods.
    :param global_kwargs: Further arguments to be passed to all methods instantiated.
    :param active_valuation_methods: A list of valuation methods to be used. If None,
        all passed valuation methods are used.
    :param kwargs_key: Key to be used for the valuation methods to define further
        arguments.
    :returns: Dictionary mapping valuation method names to functions.
    """

    if active_valuation_methods is not None:
        valuation_method_configs = {
            k: valuation_method_configs[k] for k in active_valuation_methods
        }

    function = dict()
    for valuation_method_name, config in valuation_method_configs.items():
        algorithm_name = config["algorithm"]
        function[valuation_method_name] = partial(
            compute_values,
            valuation_method=algorithm_name,
            **(config[kwargs_key] if kwargs_key in config else {}),
            **(global_kwargs if global_kwargs else {}),
        )

    return function
