from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

__all__ = [
    "plot_values_histogram",
]


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
        data:
        abscissa:
        mean_color:
        shade_color:
        ax:
        n_bootstrap_samples:
        **kwargs:

    Returns:

    """
    assert len(data.shape) == 2
    mean = data.mean(axis=1)
    sampled_idx = np.random.choice(range(data.shape[1]), n_bootstrap_samples)
    sampled_data = data.iloc[:, sampled_idx]  # TODO

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


import seaborn as sns


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
