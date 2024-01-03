"""
Calculate in-class and out-of-class marginal accuracies. Consider an arbitrary dataset D. Each point in the dataset has
a positive or negative effect onto the prediction quality of a model M. This influence can be further subdivided into
two independent utilities. The first one measures the influence on other samples of the same class label. While the
second one measures the influence onto the complement of all samples of the same class. Both of them are then used to
group all data points into four different categories. All categories depend on a threshold lambda. This parameter is
varying and the relative percentage of data points is plotted, having

1. Improves in-class accuracy and decreases out-of-class accuracy.
2. Improves in-of-class accuracy and increases out-of-class accuracy.
3. Decreases in-of-class accuracy and decreases out-of-class accuracy.
4. Decreases in-of-class accuracy and increases out-of-class accuracy.

Furthermore, the x-axis is cut such that mostly all values displayer are bigger than 0.
"""
import json
import os

import click
import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Scorer, SupervisedModel

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import plot_threshold_characteristics
from re_classwise_shapley.utils import (
    calculate_threshold_characteristic_curves,
    load_params_fast,
    pipeline_seed,
)
from re_classwise_shapley.valuation_methods import calculate_subset_score

logger = setup_logger("determine_in_out_of_clas_accuracy")


class SubsetScorer(Scorer):
    """
    A scorer which operates on a subset and additionally normalizes the output score.

    Args:
        subset: An array of indices mapping to the subset of training indices to include in the score calculation.
        normalize: True, iff the score shall be multiplied by `len(subset) / len(train_indices)`.
    """

    def __init__(
        self, *args, subset: NDArray[np.int_], normalize: bool = True, **kwargs
    ):
        Scorer.__init__(self, *args, **kwargs)
        self._idx = subset
        self._normalize = normalize

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        n = len(y)
        idx = self._idx
        score = Scorer.__call__(self, model=model, X=X[idx], y=y[idx])
        return score * len(idx) / n


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def determine_in_cls_out_of_cls_marginal_accuracies(
    experiment_name: str,
    model_name: str,
):
    """
    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        model_name: Model to use. As specified in the `params.models` section.
    """
    _determine_in_cls_out_of_cls_marginal_accuracies(experiment_name, model_name)


def _determine_in_cls_out_of_cls_marginal_accuracies(
    experiment_name: str,
    model_name: str = "logistic_regression",
    valuation_method_name: str = "tmc_shapley",
    max_plotting_percentage: float = 1e-4,
):
    output_dir = Accessor.INFO_PATH / experiment_name
    if os.path.exists(output_dir / "threshold_characteristics.svg") and os.path.exists(
        output_dir / "characteristics.json"
    ):
        return logger.info(f"Plot exist in '{output_dir}'. Skipping...")

    params = load_params_fast()
    results = {}

    dataset_names = params["datasets"].keys()
    seed = pipeline_seed(42, 8)
    seed_seqs = np.random.SeedSequence(seed).spawn(len(dataset_names))

    for idx_dataset, dataset_name in enumerate(dataset_names):
        logger.info(f"Processing dataset {dataset_name}")
        val_set = Accessor.datasets(experiment_name, dataset_name).loc[0, "val_set"]
        seed_seq_dataset = seed_seqs[idx_dataset]

        params = load_params_fast()
        backend = params["settings"]["backend"]
        n_jobs = params["settings"]["n_jobs"]
        model_seed, sampler_seed = tuple(seed_seq_dataset.generate_state(2))

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

        results[dataset_name] = {
            "in_cls_stats": in_cls_stats,
            "out_of_cls_stats": out_of_cls_stats,
            "threshold_characteristics": calculate_threshold_characteristic_curves(
                in_cls_mar_acc, out_of_cls_mar_acc
            ),
        }

    os.makedirs(output_dir, exist_ok=True)
    fig = plot_threshold_characteristics(
        results, max_plotting_percentage=max_plotting_percentage
    )
    fig.savefig(output_dir / "threshold_characteristics.svg")

    with open(output_dir / "characteristics.json", "w") as f:
        json.dump(
            results,
            f,
            sort_keys=True,
            indent=4,
        )


if __name__ == "__main__":
    determine_in_cls_out_of_cls_marginal_accuracies()
