import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from render_plots import COLOR_ENCODING, COLORS, logger

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.io import save_df_as_table
from re_classwise_shapley.utils import load_params_fast

__all__ = [
    "shaded_mean_normal_confidence_interval",
    "plot_histogram",
    "plot_metric_table",
    "plot_curves",
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


def plot_histogram(
    experiment_name: str,
    model_name: str,
    output_folder: Path,
    patch_size: Tuple[float, float] = (4, 4),
):
    """
    Plot the histogram of the data values for each dataset and valuation method.

    Args:
        experiment_name: Experiment name to obtain histograms of.
        model_name: Model name to obtain histograms of.
        output_folder: Folder to store the plots in.
        patch_size: Size of one image patch of the multi plot.
    """
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    repetitions = params_active["repetitions"]

    accessor = Accessor(experiment_name, model_name)
    valuation_results = accessor.valuation_results(
        dataset_names, valuation_methods, repetitions
    )

    output_folder = output_folder / "densities"
    h = 3
    w = int((len(valuation_results) + h - 1) / h)
    fig_ax_d = {}
    idx = 0

    for dataset_name, dataset_config in valuation_results.items():
        for method_name, method_values in dataset_config.items():
            logger.info(f"Plotting histogram for {dataset_name=}, {method_name=}.")

            if method_name not in fig_ax_d.keys():
                fig, ax = plt.subplots(
                    w, h, figsize=(w * patch_size[0], h * patch_size[1])
                )
                ax = ax.flatten()

                fig_ax_d[method_name] = (fig, ax)

            fig, ax = fig_ax_d[method_name]
            sns.histplot(
                method_values.reshape(-1),
                kde=True,
                ax=ax[idx],
                bins="sturges",
                alpha=0.4,
                color=COLORS[COLOR_ENCODING[method_name]][0],
            )
            if idx % h != 0:
                ax[idx].set_ylabel("")

            ax[idx].set_title(
                f"({chr(97 + idx)}) {dataset_name}",
                color="black",
            )

        idx += 1

    for key, (fig, ax) in fig_ax_d.items():
        for i in range(len(ax)):
            ax[i].xaxis.set_ticks(np.linspace(*ax[i].get_xlim(), 5))
            ax[i].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        for i in range(len(ax)):
            ax[i].patch.set_facecolor("none")
            ax[i].patch.set_alpha(0.0)

        f_name = f"density.{key}.svg"
        logger.info(f"Logging plot '{f_name}'")
        os.makedirs(output_folder, exist_ok=True)
        output_file = output_folder / f_name

        plt.subplots_adjust(
            left=0.05, right=0.98, top=0.97, bottom=0.03, hspace=0.17, wspace=0.17
        )
        fig.savefig(output_file, transparent=True)
        mlflow.log_artifact(str(output_file), f"densities")
        plt.close(fig)


def plot_curves(
    results_per_dataset: Dict[
        str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]
    ],
    output_folder: Path,
    patch_size: Tuple[float, float] = (4, 3),
):
    """
    Plot the curves of the data values for each dataset and valuation method.

    Args:
        results_per_dataset: A dictionary of dictionaries containing realizations of
            the distribution over curves.
        output_folder: Output folder to store the plots in.
        patch_size: Size of one image patch of the multi plot.
    """
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    metric_names = sorted(
        results_per_dataset[dataset_names[0]][valuation_methods[0]].keys()
    )
    output_folder = output_folder / "curves"
    os.makedirs(output_folder, exist_ok=True)
    num_datasets = len(dataset_names)
    w = 3
    h = int((len(dataset_names) + w - 1) / w)
    for metric_name in metric_names:
        fig, ax = plt.subplots(h, w, figsize=(h * patch_size[0], w * patch_size[1]))
        ax = ax.flatten()

        for idx, dataset_name in enumerate(dataset_names):
            for method_name in valuation_methods:
                logger.info(
                    f"Plotting curve for {dataset_name=}, {metric_name=}, {method_name=}."
                )
                results = results_per_dataset[dataset_name][method_name][metric_name][1]
                color_name = COLOR_ENCODING[method_name]
                mean_color, shade_color = COLORS[color_name]
                results = results.sort_index()

                shaded_mean_normal_confidence_interval(
                    results,
                    abscissa=results.index,
                    mean_color=mean_color,
                    shade_color=shade_color,
                    label=method_name,
                    ax=ax[idx],
                )
                ax[idx].set_title(
                    f"({chr(97 + idx)}) {dataset_name}",
                    color="black",
                )

        handles, labels = ax[num_datasets - 1].get_legend_handles_labels()
        for i in range(num_datasets, len(ax)):
            ax[i].grid(False)
            ax[i].axis("off")
            ax[i].patch.set_facecolor("none")
            ax[i].patch.set_alpha(0.0)

        legend_kwargs = {"framealpha": 0}

        leg = fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=5,
            fontsize=12,
            fancybox=False,
            shadow=False,
            **legend_kwargs,
        )

        for text in leg.get_texts():
            plt.setp(text, color="k")

        plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.13, hspace=0.3)
        os.makedirs(output_folder, exist_ok=True)
        local_path = output_folder / f"{metric_name}.svg"
        fig.savefig(local_path, transparent=True)
        mlflow.log_artifact(str(local_path), f"curves")
        plt.close(fig)


def plot_metric_table(
    results_per_dataset: Dict[
        str, Dict[str, Dict[str, Tuple[pd.Series, pd.DataFrame]]]
    ],
    output_folder: Path,
):
    """
    Renders the mean, standard deviation and coefficient of variation of the metrics.

    Args:
        results_per_dataset: A dictionary of dictionaries containing realizations of
            the distribution over metrics.
        output_folder: Output folder to store the plots in.
    """
    params = load_params_fast()
    params_active = params["active"]
    dataset_names = params_active["datasets"]
    valuation_methods = params_active["valuation_methods"]
    metric_names = sorted(
        results_per_dataset[dataset_names[0]][valuation_methods[0]].keys()
    )
    output_folder = output_folder / "metrics"
    os.makedirs(output_folder, exist_ok=True)

    for metric_name in metric_names:
        mean_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)
        std_metric = pd.DataFrame(index=dataset_names, columns=valuation_methods)

        for dataset_name in dataset_names:
            for method_name in valuation_methods:
                logger.info(
                    f"Plotting metric for {dataset_name=}, {metric_name=}, {method_name=}."
                )
                val = results_per_dataset[dataset_name][method_name][metric_name][
                    0
                ].loc[metric_name]
                m = np.mean(val)
                s = np.std(val)
                mean_metric.loc[dataset_name, method_name] = m
                std_metric.loc[dataset_name, method_name] = s
                mlflow.log_metric(f"{metric_name}_{dataset_name}_{method_name}_mean", m)
                mlflow.log_metric(f"{metric_name}_{dataset_name}_{method_name}_std", s)

        local_path = output_folder / f"{metric_name}_mean.svg"
        save_df_as_table(mean_metric.astype(float), local_path)
        mlflow.log_artifact(str(local_path), "metrics")
        mlflow.log_text(mean_metric.to_markdown(), f"metrics/{metric_name}_mean.md")

        local_path = output_folder / f"{metric_name}_std.svg"
        save_df_as_table(std_metric.astype(float), local_path)
        mlflow.log_artifact(str(local_path), "metrics")
        mlflow.log_text(std_metric.to_markdown(), f"metrics/{metric_name}_std.md")

        local_path = output_folder / f"{metric_name}_cv.svg"
        save_df_as_table(
            std_metric.astype(float) / mean_metric.astype(float), local_path
        )
        mlflow.log_artifact(str(local_path), "metrics")
        mlflow.log_text(std_metric.to_markdown(), f"metrics/{metric_name}_cv.md")
