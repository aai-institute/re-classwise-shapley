from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

__all__ = [
    "setup_plotting",
    "plot_utility_over_removal_percentages",
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
    values_df: pd.DataFrame,
    method_names: list[str],
    hue_column: str,
    *,
    output_dir: Path,
) -> None:
    colors = ["dodgerblue", "darkorange", "limegreen", "indianred", "darkorchid"]
    palette = {
        value: color for value, color in zip(values_df[hue_column].unique(), colors)
    }

    values_df = values_df.groupby(["method", hue_column]).mean(numeric_only=True)

    for method_name in method_names:
        fig, ax = plt.subplots()
        df = values_df.loc[method_name].reset_index()
        df = pd.melt(df, id_vars=[hue_column])

        sns.histplot(
            data=df,
            x="value",
            hue=hue_column,
            multiple="layer",
            kde=True,
            palette=palette,
            ax=ax,
        )
        plt.legend()
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=5,
            title=hue_column.replace("_", " ").capitalize(),
            frameon=False,
        )
        ax.set_xlabel("Value")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"values_histogram_{method_name=}.pdf",
            bbox_inches="tight",
        )


def plot_utility_over_removal_percentages(
    scores_df: pd.DataFrame,
    *,
    budgets: list[int],
    method_names: list[str],
    removal_percentages: list[float],
    output_dir: Path,
) -> None:
    mean_colors = ["dodgerblue", "darkorange", "limegreen", "indianred", "darkorchid"]
    shade_colors = ["lightskyblue", "gold", "seagreen", "firebrick", "plum"]

    for budget in budgets:
        for type in ["best", "worst"]:
            fig, ax = plt.subplots()
            for i, method_name in enumerate(method_names):
                df = scores_df.query(
                    "(method == @method_name) & (type == @type) & (budget == @budget)"
                ).drop(columns=["method", "budget", "type"], errors="ignore")

                shaded_mean_normal_confidence_interval(
                    df,
                    abscissa=removal_percentages,
                    mean_color=mean_colors[i],
                    shade_color=shade_colors[i],
                    xlabel="Percentage Removal",
                    ylabel="Accuracy",
                    label=f"{method_name}",
                    ax=ax,
                )
            plt.legend(loc="lower left")
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=3,
                title="Method",
                frameon=False,
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / f"utility_over_removal_percentages_{type=}_{budget=}.pdf",
                bbox_inches="tight",
            )
