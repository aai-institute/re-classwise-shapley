import math as m
from contextlib import contextmanager
from typing import Any, Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import OneOrMany, ensure_list

__all__ = [
    "plot_histogram",
    "plot_curves",
    "plot_time",
    "plot_metric_boxplot",
    "plot_metric_table",
]

logger = setup_logger(__name__)


# Mapping from method names to single colors
COLOR_ENCODING = {
    "RND": "black",
    "BETS": "blue",
    "LOO": "orange",
    "TMCS": "green",
    "CWS": "red",
    "OWS": "purple",
    "BZHS": "turquoise",
    "LC": "gray",
}

# Mapping from colors to mean and shade color.
COLORS = {
    "black": ("black", "silver"),
    "blue": ("dodgerblue", "lightskyblue"),
    "orange": ("darkorange", "gold"),
    "green": ("limegreen", "seagreen"),
    "red": ("indianred", "firebrick"),
    "purple": ("darkorchid", "plum"),
    "gray": ("gray", "lightgray"),
    "turquoise": ("turquoise", "lightcyan"),
}


LABELS = {
    "random": "RND",
    "beta_shapley": "BETS",
    "loo": "LOO",
    "tmc_shapley": "TMCS",
    "classwise_shapley": "CWS",
    "owen_sampling_shapley": "OWS",
    "banzhaf_shapley": "BZHS",
    "least_core": "LC",
}


def shaded_mean_normal_confidence_interval(
    data: pd.DataFrame,
    abscissa: Sequence[Any] = None,
    mean_color: str = "dodgerblue",
    shade_color: str = "lightblue",
    ax: Axes = None,
    n_bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    **kwargs,
):
    """
    Plots the mean of a dataframe with a shaded confidence interval on the given axis.

    Args:
        data: Dataframe to plot.
        abscissa: Abscissa to plot the data on. The abscissa represents the x-axis.
        mean_color: Color of the mean line.
        shade_color: Color of the shaded confidence interval.
        ax: Axis to plot on.
        n_bootstrap_samples: Number of bootstrap samples to use for the confidence
            interval.
        confidence:  Size of the confidence interval.
    """
    assert len(data.shape) == 2
    mean = data.mean(axis=1)
    sampled_idx = np.random.choice(range(data.shape[1]), n_bootstrap_samples)
    sampled_data = data.iloc[:, sampled_idx]

    no_confidence = 1 - confidence
    upper_bound = np.quantile(sampled_data, q=1 - no_confidence / 2, axis=1)
    lower_bound = np.quantile(sampled_data, q=no_confidence / 2, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    ax.fill_between(
        abscissa,
        np.minimum(upper_bound, 1.0),
        lower_bound,
        alpha=0.3,
        color=shade_color,
    )
    ax.plot(abscissa, mean, color=mean_color, **kwargs)


@contextmanager
def plot_grid_over_datasets(
    data: pd.DataFrame,
    plot_func: Callable,
    patch_size: Tuple[float, float] = (4, 4),
    n_cols: int = 3,
    legend: bool = False,
    format_x_ticks: str = None,
    tick_params_left_only: bool = False,
    tick_params_below_only: bool = False,
    grid: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
) -> plt.Figure:
    """
    Generalized function for plotting data using a specified plot function.

    Args:
        data: A pd.DataFrame containing columns `time_s`, `dataset_name`
            and `method_name`.
        plot_func: A callable function for plotting data.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.
        legend: True, if a legend should be plotted below and outside the grid of
            subplots.
        format_x_ticks: If not None, it defines the format of the x ticks.
        tick_params_below_only: If True, only the x ticks below the plot are shown.
        tick_params_left_only: If True, only the y ticks left of the plot are shown.
        grid: True, iff a grid should be displayed for guidance.
        **kwargs: Additional keyword arguments to pass to the plot_func.

    Returns:
        A figure containing the plot.
    """
    dataset_names = data["dataset_name"].unique().tolist()
    n_rows = int((len(dataset_names) + n_cols - 1) / n_cols)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_rows * patch_size[0], n_cols * patch_size[1])
    )
    ax = ax.flatten()

    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset_data = data.loc[data["dataset_name"] == dataset_name].copy()
        plot_func(data=dataset_data, ax=ax[dataset_idx], **kwargs)

        ax[dataset_idx].patch.set_facecolor("none")
        ax[dataset_idx].patch.set_alpha(0.0)

        if dataset_idx % n_cols != 0:
            ax[dataset_idx].set_ylabel("")
            if tick_params_left_only:
                ax[dataset_idx].tick_params(labelleft=False, left=False)
        else:
            ax[dataset_idx].set_ylabel(ylabel)

        if int(dataset_idx / n_cols) < n_rows - 1:
            ax[dataset_idx].set_xlabel("")
            if tick_params_below_only:
                ax[dataset_idx].tick_params(labelbottom=False, bottom=False)
        else:
            ax[dataset_idx].set_xlabel(xlabel)

        if format_x_ticks is not None:
            ax[dataset_idx].xaxis.set_ticks(np.linspace(*ax[dataset_idx].get_xlim(), 5))
            ax[dataset_idx].xaxis.set_major_formatter(
                FormatStrFormatter(format_x_ticks)
            )
        ax[dataset_idx].set_title(f"({chr(97 + dataset_idx)}) {dataset_name}")
        if grid:
            ax[dataset_idx].grid(True)

    plt.tight_layout()
    if legend:
        legend_kwargs = {"framealpha": 0}
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=5,
            fontsize=12,
            fancybox=False,
            shadow=False,
            **legend_kwargs,
        )
        fig.subplots_adjust(bottom=0.1)

    yield fig
    plt.close(fig)


@contextmanager
def plot_histogram(
    data: pd.DataFrame,
    method_names: OneOrMany[str],
    patch_size: Tuple[float, float] = (4, 4),
    n_cols: int = 3,
) -> plt.Figure:
    """
    Plot the histogram of the data values for each dataset and valuation method.

    Args:
        data: A pd.DataFrame containing columns `dataset_name` and `method_name`.
        method_names: A list of method names to plot.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.

    Returns:
        A figure containing the plot.
    """

    def plot_histogram_func(
        data: pd.DataFrame, ax: plt.Axes, method_names: List[str], **kwargs
    ):
        """
        Plot a histogram for each valuation_method.
        Args:
            data: A pd.DataFrame containing columns `dataset_name` and `method_name`.
            ax: Axes to plot the subplot on.
            method_names: A list of method names to plot.
            **kwargs:

        Returns:

        """
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        for method_name in method_names:
            method_dataset_valuation_results = data.loc[
                data["method_name"] == LABELS[method_name]
            ]
            method_values = np.stack(
                method_dataset_valuation_results["valuation"].apply(lambda v: v.values)
            )
            sns.histplot(
                method_values.reshape(-1),
                kde=True,
                ax=ax,
                bins="sturges",
                alpha=0.4,
                color=COLORS[COLOR_ENCODING[LABELS[method_name]]][0],
                label=LABELS[method_name],
            )

    with plot_grid_over_datasets(
        data,
        plot_histogram_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=True,
        method_names=ensure_list(method_names),
        xlabel="Value",
        ylabel="counts",
        format_x_ticks="%.3f",
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_time(
    data: pd.DataFrame,
    patch_size: Tuple[float, float] = (4, 4),
    n_cols: int = 3,
) -> plt.Figure:
    """
    Plot execution times as boxplot.

    Args:
        data: A pd.DataFrame containing columns `time_s`, `dataset_name`
            and `method_name`.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.

    Returns:
        A figure containing the plot.
    """

    def plot_time_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        sns.boxplot(
            data=data,
            x="time_s",
            y="method_name",
            palette=COLOR_ENCODING,
            width=0.5,
            ax=ax,
        )

    with plot_grid_over_datasets(
        data,
        plot_time_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=False,
        xlabel="s",
        ylabel="",
        tick_params_left_only=True,
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_curves(
    data: pd.DataFrame,
    patch_size: Tuple[float, float] = (4, 3),
    n_cols: int = 3,
    len_curve_perc: float = None,
) -> plt.Figure:
    """
    Plot the curves of the data values for each dataset and valuation method.

    Args:
        data: A pd.DataFrame with the curve data.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.
        len_curve_perc: Percentage of the curve length to plot.
    """

    def plot_curves_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        for method_name, method_data in data.groupby("method_name"):
            color_name = COLOR_ENCODING[method_name]
            mean_color, shade_color = COLORS[color_name]

            results = pd.concat(method_data["curve"].tolist(), axis=1)
            if len_curve_perc is not None:
                results = results.iloc[
                    : int(m.ceil(len_curve_perc * results.shape[0])), :
                ]

            shaded_mean_normal_confidence_interval(
                results,
                abscissa=results.index,
                mean_color=mean_color,
                shade_color=shade_color,
                label=method_name,
                ax=ax,
            )

    with plot_grid_over_datasets(
        data,
        plot_curves_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=True,
        xlabel="s",
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_metric_table(
    data: pd.DataFrame,
) -> plt.Figure:
    """
    Takes a pd.DataFrame and plots it as a table.
    """
    data.columns = [LABELS[c] for c in data.columns]
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, cmap=plt.cm.get_cmap("viridis"), ax=ax)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    yield fig
    plt.close(fig)


@contextmanager
def plot_metric_boxplot(
    data: pd.DataFrame,
    patch_size: Tuple[float, float] = (4, 4),
    n_cols: int = 3,
) -> plt.Figure:
    """
    Takes a linear pd.DataFrame and creates a table for it, while red

    Args:
        data: Expects a pd.DataFrame with columns specified by col_index, col_columns
            and col_cell.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.
    """

    def plot_metric_boxplot_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        sns.boxplot(
            data=data,
            x="metric",
            y="method_name",
            palette=COLOR_ENCODING,
            width=0.5,
            ax=ax,
        )

    with plot_grid_over_datasets(
        data,
        plot_metric_boxplot_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=False,
        xlabel="s",
        ylabel="",
        tick_params_left_only=True,
        grid=True,
    ) as fig:
        yield fig
