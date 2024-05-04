"""
FIXME: This file is a hack with tons of copypasta. It should be included into
render_plots.py
"""

from __future__ import annotations
from typing import Sequence

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from pydvl.value import ValuationResult
from pydvl.reporting.plots import plot_ci_array
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from re_classwise_shapley.plotting import COLORS, COLOR_ENCODING, LABELS
from re_classwise_shapley.utils import load_params_fast

params = load_params_fast()
active_params = params["active"]

ALL_METHODS = active_params["methods"]
ALL_DATASETS = active_params["datasets"]
ALL_MODELS = active_params["models"]

PLOT_FORMAT = params["settings"]["plot_format"]
BASE_PATH = "../output"
SOURCE_MODEL = "logistic_regression"
N_RUNS = len(active_params["repetitions"])


def varwad(model: str, base_path: str):
    """ """

    def read_csv(model: str, method: str, dataset: str) -> list[pd.DataFrame]:
        acc = []
        for run in range(1, N_RUNS + 1):
            df = pd.read_csv(
                f"{base_path}/{model}/{dataset}/{run}/"
                f"{method}/accuracy_{model}.csv"
            )
            df["run"] = run
            df = df.rename(columns={"n_points_removed": "t"}).sort_values(
                by="t", ascending=True
            )
            acc.append(df)
        return acc

    def compute_varwad(
        result: pd.DataFrame,
        std_correction: NDArray,
        mean_random: NDArray,
    ) -> float:
        n = max(len(result["accuracy"]), len(mean_random), len(std_correction))

        def pad_array(a: NDArray, default: float):
            if len(a) < n:
                a = np.pad(a, (0, n - len(a)), mode="constant", constant_values=default)
            return a

        a = pad_array(result["accuracy"].values, default=0.0)
        m = pad_array(mean_random, default=0.0)
        s = pad_array(std_correction, default=min(std_correction))
        # weights = np.exp(-lambda_ * np.arange(n)) * (1 - s)
        weights = (1 - s) / np.arange(1, len(s) + 1)
        return weights.dot(m - a).mean()  # type: ignore

    methods = set(ALL_METHODS).difference(["random"])
    varwad = {}
    for dataset in ALL_DATASETS:
        random = pd.concat(read_csv(model, "random", dataset))
        random = random.groupby("t").agg({"accuracy": ["mean"]}).reset_index()
        random.columns = ["t", "mean_accuracy"]
        random = random["mean_accuracy"].values

        varwad[dataset] = {k: np.zeros(N_RUNS) for k in methods}

        for method in methods:
            results = read_csv(model, method, dataset)
            r = pd.concat(results)
            r = r.groupby("t").agg({"accuracy": ["std"]}).reset_index()
            r = r.reset_index()
            r.columns = ["t", "accuracy", "std"]
            std_correction = r["std"].values / (r["std"].max() or 1)
            for p in range(N_RUNS):
                varwad[dataset][method][p] = (
                    compute_varwad(results[p], std_correction, random) / N_RUNS
                )

    return varwad


def plot_varwad(
    model: str,
    base_path: str,
    n_bootstrap_samples: int,
    title: bool = False,
):
    varwads = varwad(
        model=model,
        base_path=base_path,
    )

    n_columns = 5
    n_rows = 2
    assert n_columns * n_rows > len(ALL_DATASETS)

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 5))
    axs = axs.flatten()
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3
    )

    from collections import OrderedDict

    methods = sorted(set(ALL_METHODS).difference(["random"]))
    palette = {LABELS[m]: COLORS[COLOR_ENCODING[LABELS[m]]][0] for m in methods}
    for plot_n, dataset in enumerate(ALL_DATASETS):
        xx = pd.DataFrame(
            OrderedDict((LABELS[m], varwads[dataset][m]) for m in methods)
        )
        sns.boxplot(
            data=xx,
            orient="h",
            ax=axs[plot_n],
            bootstrap=n_bootstrap_samples,
            palette=palette,
            showfliers=False,
        )
        axs[plot_n].set_title(dataset)
        axs[plot_n].set_yticklabels([])
        axs[plot_n].tick_params(axis="y", length=0)
        axs[plot_n].grid(True)

    m = len(axs)
    for ax in axs[plot_n + 1 : m - 1]:
        ax.remove()

    last = axs[m - 1]
    import matplotlib.patches as mpatches

    legend_patches = [
        mpatches.Patch(
            color=COLORS[COLOR_ENCODING[LABELS[method]]][0],
            label=LABELS[method],
        )
        for method in methods
    ]

    last.legend(
        handles=legend_patches, loc="center", prop={"size": 9}
    )  # size in points
    last.set_axis_off()

    if title:
        fig.text(
            0.5,
            1.0,
            f"VarWAD, $\lambda = {lambda_:.02f}$",
            ha="center",
            va="top",
        )

    return fig


###############################################################################
# Plot of value decay


def plot_value_decay(
    model: str, base_path: str, n_columns: int, n_rows: int, fraction: float
):
    assert n_columns * n_rows > len(ALL_DATASETS)

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 5))
    axs = axs.flatten()
    plt.subplots_adjust(
        left=0.055, right=0.95, top=0.95, bottom=0.12, wspace=0.3, hspace=0.3
    )
    fig.text(0.02, 0.5, "Normalized value", va="center", rotation="vertical")
    fig.text(0.5, 0.02, "Value rank", ha="center")

    plot_n = 0

    methods = sorted(set(ALL_METHODS).difference(["random"]))
    for dataset in ALL_DATASETS:
        for method in methods:
            m = -1
            vals = None
            for n in range(1, N_RUNS + 1):
                with open(
                    f"{base_path}/{model}/{dataset}/{n}/valuation." f"{method}.pkl",
                    "rb",
                ) as f:
                    values: ValuationResult = pickle.load(f)
                values.sort(reverse=True)
                if vals is None:
                    m = int(len(values) * fraction)
                    vals = np.zeros((N_RUNS, m))
                vals[n - 1] = values.values[:m] / max(abs(values.values))
            axs[plot_n].set_title(f"{dataset}")
            axs[plot_n].set_xticks(np.linspace(0, m, 5, dtype=int))
            axs[plot_n].set_xticklabels(np.linspace(0, m, 5, dtype=int))
            axs[plot_n].set_xlim((0, m))
            axs[plot_n].grid(True)
            mean_color, shade_color = COLORS[COLOR_ENCODING[LABELS[method]]]

            plot_ci_array(
                vals,
                level=0.01,
                abscissa=list(range(vals.shape[1])),
                mean_color=mean_color,
                shade_color=shade_color,
                type="auto",
                ax=axs[plot_n],
                label=LABELS[method],
            )

        plot_n += 1

    m = len(axs)
    for ax in axs[plot_n : m - 1]:
        ax.remove()
    handles, labels = axs[0].get_legend_handles_labels()

    last = axs[m - 1]
    last.set_axis_off()
    last.legend(handles, labels, loc="center", prop={"size": 9})  # size in points

    return fig


###############################################################################
# Rank statistics


def rank_stability(model: str, base_path: str, alpha_range: Sequence):
    """
    Compute the fraction of indices that are stable across runs for each method
    and dataset.

    Args:
        model: The model for which to compute the rank stability.
        base_path: The base path to the valuation results.
        alpha_range: fraction of indices to consider. E.g. [0.01, 0.02, ...,
        0.5]
            Use negative values to indicate a fraction of indices to consider
            from the bottom, e.g. [-0.01, -0.02, ..., -0.5]

    """

    assert N_RUNS > 1
    assert np.min(alpha_range) >= 0 or np.max(alpha_range) <= 0

    def read_values(model: str, method: str, dataset: str, run: int) -> ValuationResult:
        with open(
            f"{base_path}/{model}/{dataset}/{run}/valuation.{method}.pkl",
            "rb",
        ) as f:
            values: ValuationResult = pickle.load(f)
        values.sort(reverse=np.all(alpha_range >= 0))
        return values

    def top_fraction(values: ValuationResult, alpha: float) -> NDArray:
        assert -1 <= alpha <= 1
        n = int(len(values) * abs(alpha))
        indices = values.indices[:n]
        return indices

    stable_fractions = {}
    for dataset in ALL_DATASETS:
        stable_fractions[dataset] = {}
        for method in ALL_METHODS:
            indices = read_values(model, method, dataset, 1).indices
            n_indices = len(indices)

            stable_fractions[dataset][method] = []
            for alpha in alpha_range:
                top_indices = set(indices)
                for n in range(2, N_RUNS + 1):
                    r = read_values(model, method, dataset, n)
                    top_indices = top_indices.intersection(top_fraction(r, alpha))
                stable_fractions[dataset][method].append(
                    len(top_indices) / (n_indices * abs(alpha))
                )

    return stable_fractions


def plot_rank_stability(
    model: str, base_path: str, n_columns: int, n_rows: int, alpha_range: Sequence
):
    assert n_columns * n_rows > len(ALL_DATASETS)

    rank_stabilities = rank_stability(
        model=model, base_path=base_path, alpha_range=alpha_range
    )

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 5))
    axs = axs.flatten()
    plt.subplots_adjust(
        left=0.055, right=0.95, top=0.95, bottom=0.12, wspace=0.3, hspace=0.3
    )
    fig.text(0.02, 0.5, "Stable %", va="center", rotation="vertical")
    fig.text(0.5, 0.02, "Top % considered", ha="center")

    plot_n = 0

    methods = sorted(set(ALL_METHODS).difference(["random", "loo"]))
    for dataset in ALL_DATASETS:
        for method in methods:
            mean_color, shade_color = COLORS[COLOR_ENCODING[LABELS[method]]]
            axs[plot_n].plot(
                np.abs(alpha_range),
                rank_stabilities[dataset][method],
                label=LABELS[method],
                color=mean_color,
            )
            axs[plot_n].set_title(f"{dataset}")
            axs[plot_n].set_ylim(0, 0.8)
            axs[plot_n].set_xlim((min(alpha_range), max(alpha_range)))
            axs[plot_n].set_xticks(np.linspace(min(alpha_range), max(alpha_range), 4))
            axs[plot_n].set_xticklabels(
                np.linspace(
                    100 * min(alpha_range), 100 * max(alpha_range), 4, dtype=int
                )
            )
            axs[plot_n].set_yticks(np.linspace(0, 0.8, 5))
            axs[plot_n].set_yticklabels(np.linspace(0, 80, 5, dtype=int))
            axs[plot_n].grid(True)

        #
        plot_n += 1

    m = len(axs)
    for ax in axs[plot_n : m - 1]:
        ax.remove()
    handles, labels = axs[0].get_legend_handles_labels()

    last = axs[m - 1]
    last.set_axis_off()
    last.legend(handles, labels, loc="center", prop={"size": 9})  # size in points

    return fig


if __name__ == "__main__":
    fig = plot_varwad(
        model=SOURCE_MODEL,
        base_path=f"{BASE_PATH}/curves/point_removal",
        n_bootstrap_samples=10000,
    )
    fig.savefig(
        f"{BASE_PATH}/plots/point_removal/"
        f"{SOURCE_MODEL}/boxplots/varwad-{SOURCE_MODEL}.box.{PLOT_FORMAT}"
    )
    fig.show()

    fig = plot_value_decay(
        model=SOURCE_MODEL,
        base_path=f"{BASE_PATH}/values/point_removal",
        n_columns=5,
        n_rows=2,
        fraction=1.0,
    )
    fig.savefig(
        f"{BASE_PATH}/plots/point_removal/{SOURCE_MODEL}/curves/value_decay-half.{PLOT_FORMAT}"
    )
    fig.show()

    fig = plot_rank_stability(
        model=SOURCE_MODEL,
        base_path=f"{BASE_PATH}/values/point_removal",
        n_columns=5,
        n_rows=2,
        alpha_range=np.arange(0.01, 0.51, 0.01),  # *-1,
    )
    fig.savefig(
        f"{BASE_PATH}/plots/point_removal/{SOURCE_MODEL}/curves/rank_stability.{PLOT_FORMAT}"
    )
    fig.show()
