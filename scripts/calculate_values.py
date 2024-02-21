import json
import os
import pickle
import time

import click
import numpy as np
from dvc.api import params_show
from pydvl.utils import Scorer, Utility

from re_classwise_shapley.config import Config
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.utils import get_pipeline_seed
from re_classwise_shapley.valuation_methods import compute_values


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
    Calculate data values for a specified dataset.
    :param dataset_name: Dataset to use.
    """

    input_dir = (
        Config.SAMPLED_PATH / experiment_name / dataset_name / str(repetition_id)
    )
    with open(input_dir / "val_set.pkl", "rb") as file:
        val_set = pickle.load(file)

    _n_pipeline_step = 2
    params = params_show()
    valuation_method_config = params["valuation_methods"][valuation_method]
    backend = params["settings"]["backend"]
    n_jobs = params["settings"]["n_jobs"]

    seed = get_pipeline_seed(repetition_id, 2)
    sub_seeds = np.random.SeedSequence(seed).generate_state(2)

    model_kwargs = params["models"][model_name]
    model = instantiate_model(model_name, model_kwargs, seed=int(sub_seeds[0]))
    u = Utility(
        data=val_set,
        model=model,
        scorer=Scorer("accuracy", default=0.0),
        catch_errors=False,
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
    output_dir = (
        Config.VALUES_PATH
        / experiment_name
        / model_name
        / dataset_name
        / str(repetition_id)
    )
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
