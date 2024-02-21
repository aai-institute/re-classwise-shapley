"""
Calculate in-class and out-of-class marginal accuracies. Consider an arbitrary dataset
D. Each point in the dataset has a positive or negative effect onto the prediction
quality of a model M. This influence can be further subdivided into two independent
utilities. The first one measures the influence on other samples of the same class
label. While the second one measures the influence onto the complement of all samples of
the same class. Both of them are then used to group all data points into four different
categories. All categories depend on a threshold lambda. This parameter is varying and
the relative percentage of data points is plotted, having

1. Improves in-class accuracy and decreases out-of-class accuracy.
2. Improves in-of-class accuracy and increases out-of-class accuracy.
3. Decreases in-of-class accuracy and decreases out-of-class accuracy.
4. Decreases in-of-class accuracy and increases out-of-class accuracy.

Furthermore, the x-axis is cut such that mostly all values displayer are bigger than 0.
"""
import os
import pickle

import click
import numpy as np
import pandas as pd

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import plot_threshold_characteristics
from re_classwise_shapley.utils import (
    calculate_threshold_characteristic_curves,
    load_params_fast,
    pipeline_seed,
)
from re_classwise_shapley.valuation_methods import calculate_subset_score

logger = setup_logger("calculate_threshold_characteristics")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
def calculate_threshold_characteristics(
    experiment_name: str,
    dataset_name: str,
    repetition_id: int,
):
    """
    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
        repetition_id: Repetition id of the experiment. It is used also as a seed for
            all randomness.
    """
    _calculate_threshold_characteristics(experiment_name, dataset_name, repetition_id)


def _calculate_threshold_characteristics(
    experiment_name: str,
    dataset_name: str,
    repetition_id: int,
):
    output_dir = (
        Accessor.THRESHOLD_CHARACTERISTICS_PATH
        / experiment_name
        / dataset_name
        / str(repetition_id)
    )
    if os.path.exists(
        output_dir / "threshold_characteristics_stats.csv"
    ) and os.path.exists(output_dir / "threshold_characteristics_curves.csv"):
        return logger.info(f"Plot exist in '{output_dir}'. Skipping...")

    params = load_params_fast()
    threshold_characteristics_settings = params["settings"]["threshold_characteristics"]
    if threshold_characteristics_settings["active"].lower() == "false":
        return logger.info("Calculation was deactivated in the settings...")

    model_name = threshold_characteristics_settings["model"]
    valuation_method_name = threshold_characteristics_settings["valuation_method"]

    model_seed, sampler_seed = tuple(
        np.random.SeedSequence(repetition_id).generate_state(2)
    )

    logger.info(
        f"Calculating threshold characteristics for '{experiment_name}/{dataset_name}/{repetition_id}'"
    )
    val_set = Accessor.datasets(experiment_name, dataset_name).loc[0, "val_set"]

    params = load_params_fast()
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]

    logger.info("Calculating in class characteristics.")
    in_cls_mar_acc, in_cls_stats = calculate_subset_score(
        val_set,
        lambda c: np.argwhere(val_set.y_train == c)[:, 0],
        model_name,
        model_seed,
        sampler_seed,
        valuation_method_name,
        n_jobs,
        backend,
    )
    logger.info("Calculating out of class characteristics.")
    out_of_cls_mar_acc, out_of_cls_stats = calculate_subset_score(
        val_set,
        lambda c: np.argwhere(val_set.y_train != c)[:, 0],
        model_name,
        model_seed,
        sampler_seed,
        valuation_method_name,
        n_jobs,
        backend,
    )

    logger.info("Calculating curves and statistics.")
    threshold_characteristics_curves = calculate_threshold_characteristic_curves(
        in_cls_mar_acc, out_of_cls_mar_acc
    )
    in_cls_out_of_cls_stats = pd.DataFrame(
        [in_cls_stats, out_of_cls_stats], index=["in_cls", "out_of_cls"]
    )

    logger.info("Storing files.")
    os.makedirs(output_dir, exist_ok=True)
    in_cls_out_of_cls_stats.to_csv(output_dir / "threshold_characteristics_stats.csv")
    threshold_characteristics_curves.to_csv(
        output_dir / "threshold_characteristics_curves.csv", sep=";"
    )


if __name__ == "__main__":
    calculate_threshold_characteristics()
