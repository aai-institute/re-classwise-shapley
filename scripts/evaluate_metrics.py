import json
import logging
import os
import pickle
from functools import partial

import click
import pandas as pd
import yaml
from dvc.api import params_show
from pydvl.parallel import ParallelConfig

from re_classwise_shapley.accessor import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.metric import metric_roc_auc, metric_weighted_metric_drop
from re_classwise_shapley.utils import load_params_fast, n_threaded

logger = setup_logger(__name__)

# Type -> Metric -> Function
MetricRegistry = {
    "weighted_metric_drop": metric_weighted_metric_drop,
    "precision_recall_roc_auc": metric_roc_auc,
}


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method", type=str, required=True)
@click.option("--metric-name", type=str, required=True)
def evaluate_metrics(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method: str,
    repetition_id: int,
    metric_name: str,
):
    """
    Calculate data values for a specified dataset.
    :param dataset_name: Dataset to use.
    """

    logger.info("Loading values, test set and preprocess info.")
    input_dir = (
        Accessor.VALUES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
    )
    os.makedirs(input_dir, exist_ok=True)
    with open(
        input_dir / f"valuation.{valuation_method}.pkl",
        "rb",
    ) as file:
        values = pickle.load(file)

    data_dir = (
        Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    with open(data_dir / "test_set.pkl", "rb") as file:
        test_set = pickle.load(file)

    preprocess_info_filename = data_dir / "preprocess_info.json"
    if os.path.exists(preprocess_info_filename):
        with open(data_dir / "preprocess_info.json", "r") as file:
            preprocess_info = json.load(file)
    else:
        preprocess_info = {}

    output_dir = (
        Accessor.RESULT_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
        / valuation_method
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Creating metric...")

    params = load_params_fast()
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]
    parallel_config = ParallelConfig(
        backend=backend,
        n_cpus_local=n_jobs,
        logging_level=logging.INFO,
    )
    metrics = params["experiments"][experiment_name]["metrics"]
    metric_kwargs = metrics[metric_name]
    metric_idx = metric_kwargs.pop("idx")
    metric_fn = partial(MetricRegistry[metric_idx], **metric_kwargs)

    logger.info("Evaluating metric...")
    with n_threaded(n_threads=1):
        metric_values, metric_curve = metric_fn(
            test_set,
            values,
            preprocess_info,
            n_jobs=n_jobs,
            config=parallel_config,
            progress=True,
        )

    evaluated_metrics = pd.Series([metric_values])
    evaluated_metrics.name = "value"
    evaluated_metrics.index.name = "metric"
    evaluated_metrics.to_csv(output_dir / f"{metric_name}.csv")
    metric_curve.to_csv(output_dir / f"{metric_name}.curve.csv")


if __name__ == "__main__":
    evaluate_metrics()
