import os
from contextlib import contextmanager
from typing import Callable, Dict

import numpy as np
import pandas as pd
import yaml
from numpy._typing import NDArray
from pydvl.utils import Seed, ensure_seed_sequence

from re_classwise_shapley.log import setup_logger

logger = setup_logger()

__all__ = [
    "flatten_dict",
    "pipeline_seed",
    "load_params_fast",
    "n_threaded",
    "linear_dataframe_to_table",
]


def pipeline_seed(initial_seed: Seed, pipeline_step: int) -> int:
    """
    Get the seed for a specific pipeline step. The seed is generated from the initial
    seed and the pipeline step.

    Args:
        initial_seed: Initial seed.
        pipeline_step: Pipeline step.

    Returns:
        The seed for the given pipeline step.
    """
    return int(ensure_seed_sequence(initial_seed).generate_state(pipeline_step)[-1])


@contextmanager
def n_threaded(n_threads: int = 1) -> None:
    """
    Context manager to temporarily set the number of threads for numpy, scipy and
    pytorch. This is necessary to avoid over-subscription of threads.

    Args:
        n_threads: Number of threads to use.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    yield None
    del os.environ["OMP_NUM_THREADS"]
    del os.environ["OPENBLAS_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]
    del os.environ["VECLIB_MAXIMUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]


def flatten_dict(d: Dict, parent_key: str = "", separator: str = ".") -> Dict:
    """
    Flatten a nested dictionary. Recursively add the values under a new key that is
    constructed by concatenating the parent key and the current key with the separator.

    Args:
        d: Dictionary to flatten.
        parent_key: Parent key to use for the new key.
        separator: Separator to use for the new key.

    Returns:
        Flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, separator=separator))
        else:
            items[new_key] = v
    return items


def load_params_fast() -> Dict:
    """
    Load the parameters from the `params.yaml` file without verification. Remove this
    call if you want to use the hydra configuration system

    Returns:
        Loaded parameters as decribed in `params.yaml`.
    """
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def linear_dataframe_to_table(
    data: pd.DataFrame,
    col_index: str,
    col_columns: str,
    col_cell: str,
    reduce_fn: Callable[[NDArray[np.float_]], float],
) -> pd.DataFrame:
    """
    Takes a linear pd.DataFrame and creates a table for it, while red

    Args:
        data: Expects a pd.DataFrame with columns specified by col_index, col_columns
            and col_cell.
        col_index: Name of the column to use as index for pd.DataFrame.
        col_columns: Name of the column to use as columns for pd.DataFrame.
        col_cell: Name of the column which holds the values.
        reduce_fn: Function to reduce the array of to a single value.

    Returns:
        A pd.DataFrame with elements from col_index as index, elements from col_columns
            as columns and elements from col_cell as content.
    """
    dataset_names = data[col_index].unique().tolist()
    valuation_method_names = data[col_columns].unique().tolist()
    df = pd.DataFrame(index=dataset_names, columns=valuation_method_names, dtype=float)
    for dataset_name in dataset_names:
        for method_name in valuation_method_names:
            df.loc[dataset_name, method_name] = reduce_fn(
                data.loc[
                    (data[col_index] == dataset_name)
                    & (data[col_columns] == method_name),
                    col_cell,
                ].values
            )
    return df


def calculate_threshold_characteristic_curves(
    in_cls_mar_acc: NDArray[np.float_],
    out_of_cls_mar_acc: NDArray[np.float_],
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """
    Varies threshold and runs through both arrays and identifies how much percent of the data exceed the threshold value
    of that specific iteration. Each threshold has four values and thus four curves are present in the final data frame.
    Args:
        in_cls_mar_acc: In-class marginal accuracies.
        out_of_cls_mar_acc: Out-of-class marginal accuracies.
        n_thresholds: Number of thresholds to use for calculating the curve.

    Returns:
        A pd.DataFrame with all four characteristic curves.
    """
    max_x = np.max(np.maximum(np.abs(in_cls_mar_acc), np.abs(out_of_cls_mar_acc)))
    x_axis = np.linspace(0, max_x, n_thresholds)

    characteristics = pd.DataFrame(index=x_axis, columns=["<,<", "<,>", ">,<", ">,>"])
    n_data = len(in_cls_mar_acc)

    for i, threshold in enumerate(characteristics.index):
        characteristics.iloc[i, 0] = (
            np.sum(
                np.logical_and(
                    in_cls_mar_acc < -threshold, out_of_cls_mar_acc < -threshold
                )
            )
            / n_data
        )
        characteristics.iloc[i, 1] = (
            np.sum(
                np.logical_and(
                    in_cls_mar_acc < -threshold, out_of_cls_mar_acc > threshold
                )
            )
            / n_data
        )
        characteristics.iloc[i, 2] = (
            np.sum(
                np.logical_and(
                    in_cls_mar_acc > threshold, out_of_cls_mar_acc < -threshold
                )
            )
            / n_data
        )
        characteristics.iloc[i, 3] = (
            np.sum(
                np.logical_and(
                    in_cls_mar_acc > threshold, out_of_cls_mar_acc > threshold
                )
            )
            / n_data
        )
    return characteristics
