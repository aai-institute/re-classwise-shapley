from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from plotly.graph_objs import Figure

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
    abscissa: Sequence[Any] = None,
    mean_color: str = "dodgerblue",
    shade_color: str = "lightblue",
    label: str = None,
    fig=None,
    row: int = 1,
    col: int = 1,
    **kwargs,
):
    """
    Plot the mean line and shaded area for the confidence interval using Plotly.
    """
    assert len(data.shape) == 2
    data = data.sort_index()
    mean = data.mean(axis=1)
    upper_bound = np.quantile(data, q=0.975, axis=1)
    lower_bound = np.quantile(data, q=0.025, axis=1)

    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    # Add shaded area
    fig.add_trace(
        go.Scatter(
            x=abscissa,
            y=upper_bound,
            line=dict(color="rgba(255,255,255,0)"),
            legendgroup=label,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    rgb_tuple = tuple(int(x * 255) for x in to_rgb(shade_color))
    fig.add_trace(
        go.Scatter(
            x=abscissa,
            y=lower_bound.tolist(),
            fill="tonexty",
            fillcolor=f"rgba{rgb_tuple + (0.4,)}",
            line=dict(color="rgba(255,255,255,0)"),
            legendgroup=label,
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=abscissa,
            y=mean,
            mode="lines",
            line=dict(color=mean_color),
            legendgroup=label,
            showlegend=row == 1 and col == 1,
            name=label,
            **kwargs,
        ),
        row=row,
        col=col,
    )


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
    curves: Dict[str, Tuple[pd.DataFrame, Dict]],
    fig: Figure,
    row: int = 1,
    col: int = 1,
):
    """
    Plot a dictionary of curves.
    :param curves: A dictionary of curves, where each curve is a tuple of (results, plot_info).
    """
    colors = {
        "black": ("black", "silver"),
        "blue": ("dodgerblue", "lightskyblue"),
        "orange": ("darkorange", "gold"),
        "green": ("limegreen", "seagreen"),
        "red": ("indianred", "firebrick"),
        "purple": ("darkorchid", "plum"),
        "gray": ("gray", "lightgray"),
        "turquoise": ("turquoise", "lightcyan"),
    }

    for method_name, (results, plot_info) in curves.items():
        mean_color, shade_color = colors[plot_info["color"]]
        shaded_mean_normal_confidence_interval(
            results,
            abscissa=results.index,
            mean_color=mean_color,
            shade_color=shade_color,
            label=method_name,
            fig=fig,
            row=row,
            col=col,
        )
