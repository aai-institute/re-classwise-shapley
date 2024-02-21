from functools import partial

import click
from dvc.api import params_show

from re_classwise_shapley.config import Config
from re_classwise_shapley.experiments import run_and_store_experiment
from re_classwise_shapley.metric import roc_auc_pr_recall
from re_classwise_shapley.preprocess import flip_labels


@click.command()
@click.option("--dataset-name", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--seed", type=int, required=False)
def run_experiment_noise_removal(
    dataset_name: str,
    model_name: str,
):
    experiment_name = "noise_removal"

    def kwargs_loader(seed: int = None):
        return {
            "label_preprocessor": partial(flip_labels, perc_flip_labels=0.2, seed=seed),
            "metrics": {
                "precision_recall": roc_auc_pr_recall,
            },
        }

    params = params_show()
    run_and_store_experiment(
        experiment_name,
        dataset_name=dataset_name,
        model_name=model_name,
        output_dir=Config.RESULT_PATH / experiment_name / model_name / dataset_name,
        loader_kwargs=kwargs_loader,
        n_repetitions=params["settings"]["evaluation"]["n_repetitions"],
    )


if __name__ == "__main__":
    run_experiment_noise_removal()
