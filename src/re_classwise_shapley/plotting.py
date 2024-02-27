import math as m
from contextlib import contextmanager
from typing import Any, Callable, List, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import OneOrMany, ensure_list

__all__ = [
    "plot_histogram",
    "plot_curves",
    "plot_time",
    "plot_metric_boxplot",
    "plot_metric_table",
    "plot_threshold_characteristics",
]

from re_classwise_shapley.utils import calculate_threshold_characteristic_curves

logger = setup_logger(__name__)


# Mapping from method names to single colors
COLOR_ENCODING = {
    "Random": "black",
    "Beta Shapley": "blue",
    "Leave-One-Out": "yellow",
    "Truncated Monte-Carlo Shapley": "green",
    "Classwise Shapley": "red",
    "Owen Sampling": "purple",
    "Banzhaf Shapley": "orange",
    "Least Core": "pink",
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
    "yellow": ("gold", "lightyellow"),
    "pink": ("pink", "lightpink"),
}


LABELS = {
    "random": "Random",
    "beta_shapley": "Beta Shapley",
    "loo": "Leave-One-Out",
    "tmc_shapley": "Truncated Monte-Carlo Shapley",
    "classwise_shapley": "Classwise Shapley",
    "owen_sampling_shapley": "Owen Sampling",
    "banzhaf_shapley": "Banzhaf Shapley",
    "least_core": "Least Core",
}


def shaded_interval_line_plot(
    data: pd.DataFrame,
    abscissa: Sequence[Any] = None,
    mean_color: str = "dodgerblue",
    shade_color: str = "lightblue",
    mean_agg: Literal["mean"] = "mean",
    std_agg: Literal["bootstrap"] | None = None,
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
    match mean_agg:
        case "mean":
            mean = data.mean(axis=1)
        case "intersect":
            pass
        case _:
            raise NotImplementedError()

    upper_bound = None
    lower_bound = None
    match std_agg:
        case "bootstrap":
            sampled_idx = np.random.choice(range(data.shape[1]), n_bootstrap_samples)
            sampled_data = data.iloc[:, sampled_idx]

            no_confidence = 1 - confidence
            upper_bound = np.quantile(sampled_data, q=1 - no_confidence / 2, axis=1)
            lower_bound = np.quantile(sampled_data, q=no_confidence / 2, axis=1)
        case None:
            pass
        case _:
            raise NotImplementedError()

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    if std_agg is not None:
        ax.fill_between(
            abscissa,
            upper_bound,
            lower_bound,
            alpha=0.3,
            color=shade_color,
        )
    ax.plot(abscissa, mean, color=mean_color, **kwargs)


@contextmanager
def plot_grid_over_datasets(
    data: pd.DataFrame,
    plot_func: Callable,
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
    legend: Union[bool, list[Patch]] = False,
    format_x_ticks: str = None,
    tick_params_left_only: bool = False,
    tick_params_below_only: bool = False,
    grid: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    x_lims: List[float] = None,
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
            subplots. Pass a list of legend handles to use a custom legend. Depending
            on the number of subplots, e.g. if it is even or odd, the legend will be
            drawn on the last filled.
        format_x_ticks: If not None, it defines the format of the x ticks.
        tick_params_below_only: If True, only the x ticks below the plot are shown.
        tick_params_left_only: If True, only the y ticks left of the plot are shown.
        grid: True, iff a grid should be displayed for guidance.
        **kwargs: Additional keyword arguments to pass to the plot_func.

    Returns:
        A figure containing the plot.
    """
    dataset_names = data["dataset_name"].unique().tolist()
    n_plots = len(dataset_names)
    n_rows = int((n_plots + n_cols - 1) / n_cols)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * patch_size[0], n_rows * patch_size[1])
    )
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3
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

        ax[dataset_idx].set_title(dataset_name)
        if grid:
            ax[dataset_idx].grid(True)

        if x_lims is not None:
            ax[dataset_idx].set_xlim([0, x_lims[dataset_idx]])

        if format_x_ticks is not None:
            ax[dataset_idx].xaxis.set_ticks(np.linspace(*ax[dataset_idx].get_xlim(), 5))
            ax[dataset_idx].xaxis.set_major_formatter(
                FormatStrFormatter(format_x_ticks)
            )

    i_first_unfilled_plot = 2 * n_plots - n_rows * n_cols + 1
    use_last_as_legend = i_first_unfilled_plot < n_rows * n_cols
    for i in range(i_first_unfilled_plot, n_rows * n_cols):
        last = ax[i]
        last.set_axis_off()

    if legend or isinstance(legend, list):
        handles_labels = ax[0].get_legend_handles_labels() if legend else (legend,)
        if use_last_as_legend:
            last = ax[-1]
            last.legend(*list(handles_labels), loc="center", prop={"size": 9})
        else:
            legend_kwargs = {"framealpha": 0}
            fig.legend(
                *list(handles_labels),
                loc="outside lower center",
                ncol=5,
                fontsize=9,
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
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
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
        ylabel="#",
        format_x_ticks="%.3f",
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_value_decay(
    data: pd.DataFrame,
    method_names: OneOrMany[str],
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
    fraction: float = 0.05,
) -> plt.Figure:
    def plot_value_decay_func(
        data: pd.DataFrame, ax: plt.Axes, method_names: List[str], **kwargs
    ):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        for method_name in method_names:
            method_dataset_valuation_results = data.loc[
                data["method_name"] == LABELS[method_name]
            ]
            method_values = np.stack(
                method_dataset_valuation_results["valuation"].apply(
                    lambda v: np.flip(np.sort(v.values))
                )
            )
            reduced_length = int(method_values.shape[1] * fraction)
            method_values = method_values[:, :reduced_length] / np.max(
                method_values, axis=1, keepdims=True
            )
            color_name = COLOR_ENCODING[LABELS[method_name]]
            mean_color, shade_color = COLORS[color_name]
            method_values = pd.DataFrame(
                method_values.T, index=np.arange(method_values.shape[1])
            )
            shaded_interval_line_plot(
                method_values,
                mean_agg="mean",
                std_agg="bootstrap",
                abscissa=method_values.index,
                mean_color=mean_color,
                shade_color=shade_color,
                label=method_name,
                ax=ax,
            )

    with plot_grid_over_datasets(
        data,
        plot_value_decay_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=True,
        method_names=ensure_list(method_names),
        xlabel="n",
        ylabel="Value",
        format_x_ticks="%.3f",
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_time(
    data: pd.DataFrame,
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
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
            hue="method_name",
            palette=COLOR_ENCODING,
            legend=False,
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
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
    mean_agg: Literal["mean"] = "mean",
    std_agg: Literal["bootstrap"] | None = None,
    plot_perc: float = None,
    x_label: str = None,
    y_label: str = None,
) -> plt.Figure:
    """
    Plot the curves of the data values for each dataset and valuation method.

    Args:
        data: A pd.DataFrame with the curve data.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.
        plot_perc: Percentage of the curve length to plot.
    """

    def plot_curves_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        for method_name, method_data in data.groupby("method_name"):
            color_name = COLOR_ENCODING[method_name]
            mean_color, shade_color = COLORS[color_name]

            results = pd.concat(method_data["curve"].tolist(), axis=1)
            results = results.iloc[1:-1]
            if plot_perc is not None:
                results = results.iloc[: int(m.ceil(plot_perc * results.shape[0])), :]

            shaded_interval_line_plot(
                results,
                mean_agg=mean_agg,
                std_agg=std_agg,
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
        xlabel=x_label,
        ylabel=y_label,
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
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
    x_label: str = None,
) -> plt.Figure:
    """
    Takes a linear pd.DataFrame and creates a table for it, while red

    Args:
        data: Expects a pd.DataFrame with columns specified by col_index, col_columns
            and col_cell.
        patch_size: Size of one image patch of the multi plot.
        n_cols: Number of columns for subplot layout.
    """

    data = data.loc[data["method_name"] != "random"]

    def plot_metric_boxplot_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        data.loc[:, "method_name"] = data["method_name"].apply(lambda m: LABELS[m])
        sns.boxplot(
            data=data,
            x="metric",
            y="method_name",
            hue="method_name",
            palette={
                label: COLORS[COLOR_ENCODING[label]][0] for label in LABELS.values()
            },
            legend=False,
            bootstrap=10000,
            width=0.5,
            ax=ax,
        )
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    legend_patches = [
        Patch(
            color=COLORS[COLOR_ENCODING[LABELS[method]]][0],
            label=LABELS[method],
        )
        for method in data["method_name"].unique()
    ]

    with plot_grid_over_datasets(
        data,
        plot_metric_boxplot_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=legend_patches,
        xlabel=x_label,
        ylabel="",
        tick_params_left_only=True,
        grid=True,
    ) as fig:
        yield fig


@contextmanager
def plot_threshold_characteristics(
    results: pd.DataFrame,
    patch_size: Tuple[float, float] = (3, 2.5),
    n_cols: int = 5,
    confidence: float = 0.95,
) -> plt.Figure:
    """
    Plots threshold characteristics for various datasets. This function takes results
    from multiple datasets and plots the threshold characteristics for each. It arranges
    the plots in a grid layout and saves the resulting figure to a specified directory.

    Args:
        results: A dictionary where each key is a dataset name and the value is another
            dictionary containing a DataFrame of threshold characteristics.
        n_cols: The number of columns in the subplot grid. Defaults to 3.
    """

    def plot_threshold_characteristics_func(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        min_threshold = 0
        max_threshold = (data.iloc[:, 1:].applymap(lambda x: np.max(x))).max().max()
        n_samples = 1000
        x_range = np.linspace(min_threshold, max_threshold, n_samples)

        n_bootstrap_samples = 1000
        all_fns_unf = [
            calculate_threshold_characteristic_curves(
                x_range, row["in_cls_mar_acc"], row["global_mar_acc"]
            )
            for _, row in data.iterrows()
        ]
        for i in range(2):
            all_fns = [df.iloc[:, i] for df in all_fns_unf]
            data = pd.concat([fn for fn in all_fns], axis=1)

            assert len(data.shape) == 2
            mean = data.mean(axis=1)
            sampled_idx = np.random.choice(range(data.shape[1]), n_bootstrap_samples)
            sampled_data = data.iloc[:, sampled_idx]

            no_confidence = 1 - confidence
            upper_bound = np.quantile(sampled_data, q=1 - no_confidence / 2, axis=1)
            lower_bound = np.quantile(sampled_data, q=no_confidence / 2, axis=1)

            mean_color, shade_color = COLORS[["green", "red"][i]]
            ax.fill_between(
                x_range,
                lower_bound.astype(float),
                upper_bound.astype(float),
                alpha=0.3,
                color=shade_color,
            )
            ax.plot(
                x_range, mean.values, label=all_fns_unf[0].columns[i], color=mean_color
            )

    with plot_grid_over_datasets(
        results,
        plot_threshold_characteristics_func,
        patch_size=patch_size,
        n_cols=n_cols,
        legend=True,
        format_x_ticks="%.5f",
        xlabel="Threshold",
        ylabel="Fraction",
        grid=True,
        x_lims=[
            0.001,
            0.00002,
            0.0025,
            0.0005,
            0.00025,
            0.0005,
            0.0005,
            0.00175,
            0.00150,
        ],
    ) as fig:
        yield fig


def plot_rank_stability(
    model: str, base_path: str, n_columns: int, n_rows: int, alpha_range: Sequence
) -> plt.Figure():
    assert n_columns * n_rows > len(ALL_DATASETS)

    rank_stabilities = rank_stability(
        model=model, base_path=base_path, alpha_range=alpha_range
    )

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(15, 5))
    axs = axs.flatten()
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.3
    )

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
