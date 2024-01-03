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
from typing import List, Union

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pydvl.utils import Scorer, SupervisedModel, Utility

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
    valuation_method_name: str = "loo",
    max_plotting_percentage: float = 1e-5,
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
        dataset_df = results[dataset_name]["detr"]
        idx = np.argwhere(
            np.max(dataset_df, axis=1) >= max_plotting_percentage, axis=1
        )[-1, 0]
        dataset_df.iloc[:idx].plot(ax=ax[dataset_idx])
        ax[dataset_idx].set_xlim(0, dataset_df.index[idx])
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


if __name__ == "__main__":
    determine_in_cls_out_of_cls_marginal_accuracies()
