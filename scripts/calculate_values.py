import json
import os
import pickle
import time

import click
import numpy as np
from pydvl.utils import Scorer, Utility

from re_classwise_shapley.accessor import Accessor
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
@click.option("--valuation-method", type=str, required=True)
def calculate_values(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    valuation_method: str,
    repetition_id: int,
):
    """
    Calculate data values for a specified dataset. The values are calculated using the
    valuation method specified in the `params.valuation_methods` section. The values
    are stored in the `Accessor.VALUES_PATH` directory.

    Args:
        experiment_name: Experiment name as specified in the `params.experiments`
            section.
        dataset_name: Dataset name as specified in the `params.datasets` section.
        model_name: Model name as specified in the `params.models` section.
        valuation_method: Valuation method name as specified in the
            `params.valuation_methods` section.
        repetition_id: Unique repetition id used for seeding the experiment
    """

    input_dir = (
        Accessor.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    output_dir = (
        Accessor.VALUES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
    )

    if os.path.exists(
        output_dir / f"valuation.{valuation_method}.pkl"
    ) and os.path.exists(output_dir / f"valuation.{valuation_method}.stats.json"):
        return logger.info(
            f"Values for {valuation_method} exist in '{output_dir}'. Skipping..."
        )

    with open(input_dir / "val_set.pkl", "rb") as file:
        val_set = pickle.load(file)

    n_pipeline_step = 2
    seed = pipeline_seed(repetition_id, n_pipeline_step)
    sub_seeds = np.random.SeedSequence(seed).generate_state(2)

    params = load_params_fast()
    valuation_method_config = params["valuation_methods"][valuation_method]
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]

    model_kwargs = params["models"][model_name]
    model = instantiate_model(model_name, model_kwargs, seed=int(sub_seeds[0]))
    u = Utility(
        data=val_set,
        model=model,
        scorer=Scorer("accuracy", default=0.0),
        catch_errors=True,
    )

    start_time = time.time()
    values = compute_values(
        u,
        valuation_method=valuation_method_config["algorithm"],
        n_jobs=n_jobs,
        backend=backend,
        seed=int(sub_seeds[1]),
        **(
            valuation_method_config["kwargs"]
            if "kwargs" in valuation_method_config
            else {}
        ),
    )
    diff_time = time.time() - start_time
    runtime_stats = {"time_s": diff_time}

    os.makedirs(output_dir, exist_ok=True)
    with open(
        output_dir / f"valuation.{valuation_method}.pkl",
        "wb",
    ) as file:
        pickle.dump(values, file)

    with open(output_dir / f"valuation.{valuation_method}.stats.json", "w") as file:
        json.dump(runtime_stats, file)


if __name__ == "__main__":
    calculate_values()
