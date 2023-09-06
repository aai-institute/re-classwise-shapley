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
def run_experiment_wad_drop(
    dataset_name: str,
    model_name: str,
):
    """
    Run an experiment and store the results of the run on disk.
    :param dataset_name: Dataset to use.
    :param model_name: Model to use.
    """
    params = params_show()
    experiment_name = "wad_drop"
    transfer_models = params["experiments"][experiment_name]["transfer_models"]

    def kwargs_loader(_seed: Seed = None):
        metrics = {}
        for transfer_model_name in transfer_models:
            transfer_model_kwargs = params["models"][transfer_model_name]
            transfer_model = instantiate_model(
                transfer_model_name, transfer_model_kwargs, seed=_seed
            )
            metrics.update(
                {  # type: ignore
                    f"highest_accuracy_drop_{transfer_model_name}": partial(
                        weighted_metric_drop,
                        highest_first=True,
                        transfer_model=transfer_model,
                        metric="accuracy",
                    ),
                    f"lowest_accuracy_drop_{transfer_model_name}": partial(
                        weighted_metric_drop,
                        highest_first=False,
                        transfer_model=transfer_model,
                        metric="accuracy",
                    ),
                    # f"highest_f1_weighted_drop_{transfer_model_name}": partial(
                    #     weighted_metric_drop,
                    #     highest_first=True,
                    #     transfer_model=transfer_model,
                    #     metric="f1_weighted",
                    # ),
                }
            )

        return {"metrics": metrics}

    output_dir = Config.RESULT_PATH / experiment_name / model_name
    output_dir /= dataset_name
    os.makedirs(output_dir, exist_ok=True)
    params = params_show()
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
