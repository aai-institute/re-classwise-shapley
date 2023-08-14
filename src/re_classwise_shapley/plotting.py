from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

__all__ = [
    "setup_plotting",
    "plot_curve",
    "plot_values_histogram",
]


def setup_plotting():
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.set_context("paper", font_scale=1.0)


def shaded_mean_normal_confidence_interval(
    data: pd.DataFrame,
    abscissa: Sequence[Any] | None = None,
    mean_color: str | None = "dodgerblue",
    shade_color: str | None = "lightblue",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Modified version of the `shaded_mean_std()` function defined in pyDVL."""
    assert len(data.shape) == 2
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    standard_error = std / np.sqrt(data.shape[0])
    upper_bound = mean + 1.96 * standard_error
    lower_bound = mean - 1.96 * standard_error

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    ax.fill_between(
        abscissa,
        upper_bound,
        lower_bound,
        alpha=0.3,
        color=shade_color,
    )
    ax.plot(abscissa, mean, color=mean_color, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_values_histogram(
    values_df: pd.DataFrame, *, ax: Axes, title: str = None
) -> None:
    df = values_df.reset_index(drop=True)
    df = df.apply(lambda s: s)

    data = np.concatenate(tuple(df.to_numpy()))
    sns.histplot(
        data=data,
        multiple="layer",
        kde=True,
        ax=ax,
    )
    if title is not None:
        ax.set_title(title, y=-0.3)

    ymin, ymax = ax.get_ylim()
    ax.vlines(np.mean(data), color="r", ymin=ymin, ymax=ymax)


def plot_curve(
    scores_df: pd.DataFrame,
    *,
    title: str = None,
    ax: Axes = None,
) -> None:
    mean_colors = ["dodgerblue", "darkorange", "limegreen", "indianred", "darkorchid"]
    shade_colors = ["lightskyblue", "gold", "seagreen", "firebrick", "plum"]
    color_pos = [
        "beta_shapley",
        "loo",
        "tmc_shapley",
        "classwise_shapley",
        "classwise_shapley_add_idx",
    ]
    color_pos = {v: i for i, v in enumerate(color_pos)}

    for i, method_name in enumerate(scores_df.columns):
        if method_name not in color_pos:
            continue

        mean_color = mean_colors[color_pos[method_name]]
        shade_color = shade_colors[color_pos[method_name]]

        scores = scores_df.loc[:, method_name].apply(lambda s: pd.Series(s))
        abscissa = list(scores.columns)
        abscissa = abscissa[: int(len(abscissa) / 2)]
        scores = scores.loc[:, abscissa]
        shaded_mean_normal_confidence_interval(
            scores,
            abscissa=abscissa,
            mean_color=mean_color,
            shade_color=shade_color,
            label=method_name,
            ax=ax,
        )
    if title is not None:
        ax.set_title(title, y=-0.25)