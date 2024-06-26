"""
Stage 4 calculates Shapley values for sampled dataset.

1. Fetch data
2. Preprocess data
3. Sample data
4. Calculate Shapley values
5. Evaluate metrics
6. Render plots

Calculates Shapley values for a sampled dataset. The sampled dataset is defined by the
experiment name, dataset name and repetition id. The Shapley values are calculated using
the valuation method specified in the `params.valuation_methods` section. The Shapley
values are stored in the `Accessor.VALUES_PATH` directory.
"""

import json
import os
import pickle
import time

import click
import numpy as np
from pydvl.utils import Scorer, Utility

from re_classwise_shapley.cache import PrefixMemcachedCacheBackend
from re_classwise_shapley.io import Accessor
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.utils import load_params_fast, pipeline_seed
from re_classwise_shapley.valuation_methods import compute_values

logger = setup_logger("calculate_values")


@click.command()
@click.option("--experiment-name", type=str, required=True)
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--repetition-id", type=int, required=True)
@click.option("--valuation-method-name", type=str, required=True)
def calculate_values(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
):
    """
    Calculate data values for a specified dataset. The values are calculated using the
    valuation method specified in the `params.valuation_methods` section. The values
    are stored in the `Accessor.VALUES_PATH` directory.

    Args:
        experiment_name: Name of the executed experiment. As specified in the
            `params.experiments` section.
        dataset_name: The name of the dataset to preprocess. As specified in th
            `params.datasets` section.
        model_name: Model to use. As specified in the `params.models` section.
        valuation_method_name: Name of the valuation method to use. As specified in the
            `params.valuation_methods` section.
        repetition_id: Repetition id of the experiment. It is used also as a seed for
            all randomness.
    """
    _calculate_values(
        experiment_name,
        dataset_name,
        model_name,
        valuation_method_name,
        repetition_id,
    )


def _calculate_values(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method_name: str,
    repetition_id: int,
):
    output_dir = (
        Accessor.VALUES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
    )

    if os.path.exists(
        output_dir / f"valuation.{valuation_method_name}.pkl"
    ) and os.path.exists(output_dir / f"valuation.{valuation_method_name}.stats.json"):
        return logger.info(
            f"Values for {valuation_method_name} exist in '{output_dir}'. Skipping..."
        )

    params = load_params_fast()
    cache = None
    if "cache_group" in params["valuation_methods"][valuation_method_name]:
        cache_group = params["valuation_methods"][valuation_method_name]["cache_group"]
        prefix = f"{experiment_name}/{dataset_name}/{model_name}/{cache_group}"
        try:
            cache = PrefixMemcachedCacheBackend(prefix=prefix)
        except ConnectionRefusedError:
            logger.info("Couldn't connect to cache backend.")
            cache = None

    val_set = Accessor.datasets(experiment_name, dataset_name).loc[0, "val_set"]

    n_pipeline_step = 4
    seed = pipeline_seed(repetition_id, n_pipeline_step)
    sub_seeds = np.random.SeedSequence(seed).generate_state(2)

    valuation_method_config = params["valuation_methods"][valuation_method_name]
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]

    model_kwargs = params["models"][model_name]
    model = instantiate_model(model_name, model_kwargs, seed=int(sub_seeds[0]))

    start_time = time.time()
    algorithm_name = valuation_method_config.pop("algorithm")
    values = compute_values(
        Utility(
            data=val_set,
            model=model,
            scorer=Scorer("accuracy", default=0.0),
            catch_errors=True,
            cache_backend=cache,
        ),
        valuation_method=algorithm_name,
        n_jobs=n_jobs,
        backend=backend,
        seed=int(sub_seeds[1]),
        **valuation_method_config,
    )
    diff_time = time.time() - start_time
    runtime_stats = {"time_s": diff_time}

    os.makedirs(output_dir, exist_ok=True)
    with open(
        output_dir / f"valuation.{valuation_method_name}.pkl",
        "wb",
    ) as file:
        pickle.dump(values, file)

    with open(
        output_dir / f"valuation.{valuation_method_name}.stats.json", "w"
    ) as file:
        json.dump(runtime_stats, file)


if __name__ == "__main__":
    calculate_values()
