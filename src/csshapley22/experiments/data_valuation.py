import click
import numpy as np
import pandas as pd
from dvc.api import params_show
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from csshapley22.constants import OUTPUT_DIR, RANDOM_SEED
from csshapley22.dataset import create_synthetic_dataset
from csshapley22.utils import (
    compute_values,
    convert_values_to_dataframe,
    set_random_seed,
    setup_logger,
)

logger = setup_logger()

set_random_seed(RANDOM_SEED)


@click.command()
@click.option(
    "--budget",
    type=int,
    help="Value computation budget i.e. number of iterations",
    required=True,
)
def run(budget: int):
    logger.info("Starting data valuation experiment")

    random_state = np.random.RandomState(RANDOM_SEED)

    params = params_show()
    logger.info(f"Using parameters:\n{params}")

    n_jobs = params["common"]["n_jobs"]
    n_repetitions = params["common"]["n_repetitions"]

    data_valuation_params = params["data_valuation"]
    dataset_type = data_valuation_params["dataset"]
    removal_percentages = data_valuation_params["removal_percentages"]
    method_names = data_valuation_params["method_names"]

    # Create the output directory
    experiment_output_dir = (
        OUTPUT_DIR
        / "data_valuation"
        / "results"
        / f"dataset={dataset_type}"
        / f"budget={budget}"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    # We do not set the random_state in the model itself
    # because we are testing the method and not the model
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

    for repetition in range(n_repetitions):
        logger.info(f"{repetition=}")

        all_values = []
        all_scores = []

        if dataset_type == "synthetic":
            n_features = data_valuation_params["n_features"]
            n_train_samples = data_valuation_params["n_train_samples"]
            n_test_samples = data_valuation_params["n_test_samples"]
            dataset = create_synthetic_dataset(
                n_features=n_features,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                random_state=random_state,
            )

        logger.info("Creating utility")
        utility = Utility(
            data=dataset,
            model=model,
            score_range=(0.0, 1.0),
        )

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

            logger.info("Computing worst data points removal score")
            scores = compute_removal_score(
                u=utility,
                values=values,
                percentages=removal_percentages,
                remove_best=False,
            )
            scores["method"] = method_name
            scores["type"] = "worst"
            all_scores.append(scores)

            logger.info("Computing best data points removal score")
            scores = compute_removal_score(
                u=utility,
                values=values,
                percentages=removal_percentages,
                remove_best=True,
            )
            scores["method"] = method_name
            scores["type"] = "best"
            all_scores.append(scores)

        logger.info("Saving results to disk")
        scores_df = pd.DataFrame(all_scores)
        scores_df.to_csv(
            experiment_output_dir / f"scores_{repetition}.csv", index=False
        )

        values_df = pd.concat(all_values)
        values_df.to_csv(
            experiment_output_dir / f"values_{repetition}.csv", index=False
        )

    logger.info("Finished data valuation experiment")


if __name__ == "__main__":
    run()
