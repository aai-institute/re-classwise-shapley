"""
Stage six renders the plots and stores them on disk and in mlflow.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Render plots for the data valuation experiment. The plots are stored in the
`Accessor.PLOT_PATH` directory. The plots are stored in `*.svg` format. The plots are
also stored in mlflow. The id of the mlflow experiment is given by the schema
`experiment_name.model_name`.
"""

import os
import os.path
import pickle
from datetime import datetime
from typing import Tuple, cast

import click
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.data.pandas_dataset import PandasDataset
from numpy.typing import NDArray
from pydvl.utils import Dataset
from pydvl.value import ValuationResult
from scipy.stats import gaussian_kde

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.io import load_results_per_dataset_and_method
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.plotting import plot_curves, plot_histogram, plot_metric_table
from re_classwise_shapley.utils import flatten_dict, load_params_fast

logger = setup_logger("render_plots")

# Mapping from method names to single colors
COLOR_ENCODING = {
    "random": "black",
    "beta_shapley": "blue",
    "loo": "orange",
    "tmc_shapley": "green",
    "classwise_shapley": "red",
    "owen_sampling_shapley": "purple",
    "banzhaf_shapley": "turquoise",
    "least_core": "gray",
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


def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Get or create a mlflow experiment. If the experiment does not exist, it will be
    created.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Identifier of the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
def render_plots(experiment_name: str, model_name: str):
    """
    Render plots for the data valuation experiment. The plots are stored in the
    `Accessor.PLOT_PATH` directory. The plots are stored in `*.svg` format.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        model_name: Model to use. As specified in the `params.models` section.
    """
    load_dotenv()
    logger.info("Starting plotting of data valuation experiment")
    experiment_path = Accessor.RESULT_PATH / experiment_name / model_name
    output_folder = Accessor.PLOT_PATH / experiment_name / model_name
    mlflow_id = f"{experiment_name}.{model_name}"

    params = load_params_fast()
    mlflow.set_tracking_uri(params["settings"]["mlflow_tracking_uri"])
    experiment_id = get_or_create_mlflow_experiment(mlflow_id)
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Starting run.")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=datetime.now().isoformat(),
    ):
        logger.info("Log params.")
        params = flatten_dict(params)
        mlflow.log_params(params)
        log_dataset(experiment_name)

        logger.info("Plotting metric tables.")
        params = load_params_fast()
        metrics = params["experiments"][experiment_name]["metrics"]
        results_per_dataset = load_results_per_dataset_and_method(
            experiment_path, metrics
        )
        plot_metric_table(results_per_dataset, output_folder)

        logger.info(f"Plotting histogram.")
        plot_histogram(experiment_name, model_name, output_folder)

        logger.info(f"Plotting curves.")
        plot_curves(results_per_dataset, output_folder)


def log_dataset(
    experiment_name: str,
):
    """
    Log the dataset as mlflow inputs.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section. The data is loaded from the
            `Accessor.SAMPLED_PATH` directory.
    """

    def _convert(x, y):
        return pd.DataFrame(np.concatenate((x, y.reshape([-1, 1])), axis=1))

    params = load_params_fast()
    for dataset_name in params["active"]["datasets"]:
        for repetition in params["active"]["repetitions"]:
            data_dir = (
                Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition)
            )

            for set_name in ["test_set", "val_set"]:
                with open(data_dir / f"{set_name}.pkl", "rb") as file:
                    test_set = cast(Dataset, pickle.load(file))

                    x = np.concatenate((test_set.x_train, test_set.x_test), axis=0)
                    y = np.concatenate((test_set.y_train, test_set.y_test), axis=0)
                    train_df = _convert(x, y)
                    train_df.columns = test_set.feature_names + test_set.target_names
                    train_dataset: PandasDataset = (
                        mlflow.data.pandas_dataset.from_pandas(
                            train_df,
                            targets=test_set.target_names[0],
                            name=f"{dataset_name}_{repetition}_{set_name}",
                        )
                    )
                    mlflow.log_input(
                        train_dataset,
                        tags={
                            "set": set_name,
                            "dataset": dataset_name,
                            "repetition": str(repetition),
                        },
                    )


def mean_density(
    values: NDArray[ValuationResult],
    hist_range: Tuple[float, float],
    bins: int = 50,
) -> NDArray[np.float_]:
    """
    Compute the mean and confidence interval of a set of `ValuationResult` objects.

    Args:
        values: List of `ValuationResult` objects.
        hist_range: Tuple of minimum hist rand and maximum hist range.
        bins: Number of bins to be used for histogram.

    Returns:
        A tuple with the mean and 95% confidence interval.
    """
    x = np.linspace(*hist_range, bins)
    kde = gaussian_kde(values.reshape(-1))
    density = kde(x)
    return density


if __name__ == "__main__":
    render_plots()
