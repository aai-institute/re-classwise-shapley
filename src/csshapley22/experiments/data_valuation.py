from pathlib import Path

import click
import pandas as pd
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Scorer, Utility
from pydvl.value.shapley.classwise import CSScorer

from csshapley22.constants import RANDOM_SEED
from csshapley22.dataset import create_diabetes_dataset, create_synthetic_dataset
from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.utils import (
    compute_values,
    convert_values_to_dataframe,
    instantiate_model,
    set_random_seed,
    setup_logger,
)

logger = setup_logger()

set_random_seed(RANDOM_SEED)


@click.command()
@click.option(
    "--dataset-name", type=str, required=True, help="Name of the dataset to use"
)
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
@click.option("--budget", type=int, required=True, help="Computation budget")
def run(dataset_name: str, model_name: str, budget: int):
    logger.info("Starting data valuation experiment")

    params = params_show()
    logger.info(f"Using parameters:\n{params}")

    n_jobs = params["common"]["n_jobs"]

    data_valuation_params = params["data_valuation"]
    n_repetitions = data_valuation_params["common"]["n_repetitions"]
    method_names = data_valuation_params["common"]["method_names"]

    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root())
        / "output"
        / "data_valuation"
        / f"dataset={dataset_name}"
        / "results"
        / f"{budget=}"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(n_repetitions):
        logger.info(f"{repetition=}")

        repetition_output_dir = experiment_output_dir / f"{repetition=}"
        repetition_output_dir.mkdir(parents=True, exist_ok=True)

        all_values = []
        all_scores = []

        if dataset_name == "diabetes":
            diabetes_dataset_params = data_valuation_params["diabetes"]
            train_size = diabetes_dataset_params["train_size"]
            dataset = create_diabetes_dataset(
                train_size=train_size,
            )
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'")

        scorer = CSScorer()
        model = instantiate_model(model_name)

        logger.info("Creating utility")
        utility = Utility(data=dataset, model=model, scorer=scorer)

        for method_name in method_names:
            logger.info(f"{method_name=}")
            logger.info("Computing values")
            values = compute_values(
                method_name, utility=utility, n_jobs=n_jobs, budget=budget
            )

            logger.info("Converting values to DataFrame")
            df = convert_values_to_dataframe(values)
            df["method"] = method_name
            all_values.append(df)

            logger.info("Computing best data points removal score")
            accuracy_utility = Utility(
                data=dataset, model=model, scorer=Scorer(scoring="accuracy")
            )
            weighted_accuracy_drop = weighted_reciprocal_diff_average(
                u=accuracy_utility, values=values, progress=True
            )
            all_scores.append(weighted_accuracy_drop)

        logger.info("Saving results to disk")
        scores_df = pd.Series(all_scores)
        scores_df.to_csv(repetition_output_dir / "scores.csv", index=False)

        values_df = pd.concat(all_values)
        values_df.to_csv(repetition_output_dir / "values.csv", index=False)

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    run()
