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
import json
import os
from typing import List, Tuple, Union

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from numpy.random import SeedSequence
from pydvl.utils import Scorer, SupervisedModel, Utility, ensure_seed_sequence

from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.types import ensure_list
from re_classwise_shapley.utils import load_params_fast, pipeline_seed
from re_classwise_shapley.valuation_methods import compute_values

logger = setup_logger("preprocess_data")


class InClsScorer(Scorer):
    """
    LOO with in-class and out-of-class predictions.
    """

    def __init__(self, *args, c_class: Union[int, List[int]], **kwargs):
        Scorer.__init__(self, *args, **kwargs)
        self.c_class = ensure_list(c_class)

    def __call__(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        n = len(y)
        idx = np.argwhere(np.isin(y, self.c_class))[:, 0]
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
        valuation_method_config = params["valuation_methods"][valuation_method_name]
        backend = params["settings"]["backend"]
        n_jobs = params["settings"]["n_jobs"]
        model_seed, sampler_seed = tuple(seed_seq_dataset.generate_state(2))
        all_classes = np.unique(val_set.y_train)
        mat_in_cls_mar_acc = []
        mat_out_of_cls_mar_acc = []

        for c in all_classes:
            # in-class accuracy
            params = load_params_fast()
            model_kwargs = params["models"][model_name]
            in_cls_u = Utility(
                data=val_set,
                model=instantiate_model(model_name, model_kwargs, seed=int(model_seed)),
                scorer=InClsScorer("accuracy", c_class=c, default=np.nan),
                catch_errors=False,
            )
            in_cls_values = compute_values(
                in_cls_u,
                valuation_method_name,
                **valuation_method_config,
                backend=backend,
                n_jobs=n_jobs,
                seed=sampler_seed,
            )
            mat_in_cls_mar_acc.append(in_cls_values.values)

            # out-of-class accuracy
            params = load_params_fast()
            model_kwargs = params["models"][model_name]
            out_of_cls_u = Utility(
                data=val_set,
                model=instantiate_model(model_name, model_kwargs, seed=int(model_seed)),
                scorer=InClsScorer(
                    "accuracy",
                    c_class=list(all_classes[all_classes != c]),
                    default=np.nan,
                ),
                catch_errors=False,
            )
            out_of_cls_values = compute_values(
                out_of_cls_u,
                valuation_method_name,
                **valuation_method_config,
                backend=backend,
                n_jobs=n_jobs,
                seed=sampler_seed,
            )
            mat_out_of_cls_mar_acc.append(out_of_cls_values.values)

        mat_in_cls_mar_acc = np.stack(mat_in_cls_mar_acc, axis=1)
        mat_out_of_cls_mar_acc = np.stack(mat_out_of_cls_mar_acc, axis=1)
        in_cls_mar_acc = np.take_along_axis(
            mat_in_cls_mar_acc, val_set.y_train.reshape([-1, 1]), axis=1
        ).reshape(-1)
        out_of_cls_mar_acc = np.take_along_axis(
            mat_out_of_cls_mar_acc, val_set.y_train.reshape([-1, 1]), axis=1
        ).reshape(-1)

        key_in_cls_acc = "in_cls_acc"
        key_out_of_cls_acc = "out_of_cls_acc"
        result = {
            key_in_cls_acc: {
                "mean": np.mean(in_cls_mar_acc),
                "std": np.std(in_cls_mar_acc),
            },
            key_out_of_cls_acc: {
                "mean": np.mean(out_of_cls_mar_acc),
                "std": np.std(out_of_cls_mar_acc),
            },
        }

        n_thresholds = 100
        max_x = np.max(np.maximum(np.abs(in_cls_mar_acc), np.abs(out_of_cls_mar_acc)))
        x = np.linspace(0, max_x, n_thresholds)
        s = pd.DataFrame(index=x, columns=["<,<", "<,>", ">,<", ">,>"])
        n = len(in_cls_mar_acc)
        for i, threshold in enumerate(s.index):
            s.iloc[i, 0] = (
                np.sum(
                    np.logical_and(
                        in_cls_mar_acc < -threshold, out_of_cls_mar_acc < -threshold
                    )
                )
                / n
            )
            s.iloc[i, 1] = (
                np.sum(
                    np.logical_and(
                        in_cls_mar_acc < -threshold, out_of_cls_mar_acc > threshold
                    )
                )
                / n
            )
            s.iloc[i, 2] = (
                np.sum(
                    np.logical_and(
                        in_cls_mar_acc > threshold, out_of_cls_mar_acc < -threshold
                    )
                )
                / n
            )
            s.iloc[i, 3] = (
                np.sum(
                    np.logical_and(
                        in_cls_mar_acc > threshold, out_of_cls_mar_acc > threshold
                    )
                )
                / n
            )

        result["detr"] = s
        results[dataset_name] = result

    os.makedirs(output_dir, exist_ok=True)
    n_cols = 3
    n_rows = int((len(dataset_names) + n_cols - 1) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_rows * 4, n_cols * 4))
    ax = ax.flatten()
    for dataset_idx, dataset_name in enumerate(dataset_names):
        results[dataset_name]["detr"].plot(ax=ax[dataset_idx])
        ax[dataset_idx].set_title(f"({chr(97 + dataset_idx)}) {dataset_name}")

    fig.suptitle("In class and out of class characteristic curves.")
    fig.show()
    fig.savefig(output_dir / "threshold_characteristics.svg")

    with open(output_dir / "characteristics.json", "w") as f:
        json.dump(
            result,
            f,
            sort_keys=True,
            indent=4,
        )


def mix_first_and_second_class_marginals(
    labels: NDArray[np.int_],
    first_class_marginals: NDArray[np.float_],
    second_class_marginals: NDArray[np.float_],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Computes the in-class and out-of-class accuracies for first and second classes.

    Args:
        labels: An array representing class labels, where 0 indicates first class and 1 indicates second class.
        first_class_marginals: An array representing marginal probabilities for the first class.
        second_class_marginals: An array representing marginal probabilities for the second class.

    Returns:
        A tuple containing two NDArrays:
            in_class_accuracies: In-class accuracies for first and second classes.
            out_of_class_accuracies: Out-of-class accuracies for first and second classes.

    """
    in_class_accuracies = np.zeros_like(first_class_marginals)
    out_of_class_accuracies = np.zeros_like(first_class_marginals)
    first_class_idx = np.argwhere(labels == 0)[:, 0]
    second_class_idx = np.argwhere(labels == 1)[:, 0]
    in_class_accuracies[first_class_idx] = first_class_marginals[first_class_idx]
    out_of_class_accuracies[first_class_idx] = second_class_marginals[first_class_idx]
    in_class_accuracies[second_class_idx] = second_class_marginals[second_class_idx]
    out_of_class_accuracies[second_class_idx] = first_class_marginals[second_class_idx]
    return in_class_accuracies, out_of_class_accuracies


if __name__ == "__main__":
    determine_in_cls_out_of_cls_marginal_accuracies()
