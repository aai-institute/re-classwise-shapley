import os
from functools import partial
from typing import Optional

import click
from dvc.api import params_show

from re_classwise_shapley.config import Config
from re_classwise_shapley.experiments import run_and_store_experiment
from re_classwise_shapley.metric import weighted_metric_drop
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.types import Seed


@click.command()
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--transfer-model-name", type=str, required=False)
def run_experiment_wad_drop(
    dataset_name: str,
    model_name: str,
    transfer_model_name: Optional[str] = None,
):
    """
    Run an experiment and store the results of the run on disk.
    :param dataset_name: Dataset to use.
    :param model_name: Model to use.
    :param transfer_model_name: Model to use for transfer learning.
    """
    experiment_name = "wad_drop"
    if transfer_model_name is not None:
        experiment_name += "_transfer"

    if transfer_model_name == model_name:
        return

    def kwargs_loader(_seed: Seed = None):
        transfer_model_low = None
        transfer_model_high = None

        if transfer_model_name is not None:
            _params = params_show()
            transfer_model_kwargs = _params["models"][transfer_model_name]
            transfer_model_low = instantiate_model(
                transfer_model_name, transfer_model_kwargs, seed=_seed
            )
            transfer_model_high = instantiate_model(
                transfer_model_name, transfer_model_kwargs, seed=_seed
            )

        return {
            "metrics": {  # type: ignore
                "highest_accuracy_drop": partial(
                    weighted_metric_drop,
                    highest_first=True,
                    transfer_model=transfer_model_high,
                    metric="accuracy",
                ),
                "lowest_accuracy_drop": partial(
                    weighted_metric_drop,
                    highest_first=False,
                    transfer_model=transfer_model_low,
                    metric="accuracy",
                ),
                "highest_f1_weighted_drop": partial(
                    weighted_metric_drop,
                    highest_first=True,
                    transfer_model=transfer_model_high,
                    metric="f1_weighted",
                ),
            }
        }

    params = params_show()
    output_dir = Config.RESULT_PATH / experiment_name / model_name
    if transfer_model_name is not None:
        output_dir /= transfer_model_name

    output_dir /= dataset_name
    os.makedirs(output_dir, exist_ok=True)
    run_and_store_experiment(
        experiment_name,
        dataset_name=dataset_name,
        model_name=model_name,
        output_dir=output_dir,
        loader_kwargs=kwargs_loader,
        n_repetitions=params["settings"]["evaluation"]["n_repetitions"],
    )


if __name__ == "__main__":
    run_experiment_wad_drop()
