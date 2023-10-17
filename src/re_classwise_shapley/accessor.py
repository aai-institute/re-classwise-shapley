import json
import pickle
from itertools import product
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from re_classwise_shapley.types import OneOrMany, ensure_list


class Accessor:
    """
    Accessor class to load data from the output directory.

    Args:
        experiment_name: Name of the executed experiment.
        model_name: Name of the model.
    """

    OUTPUT_PATH = Path("./output")
    RAW_PATH = OUTPUT_PATH / "raw"
    PREPROCESSED_PATH = OUTPUT_PATH / "preprocessed"
    SAMPLED_PATH = OUTPUT_PATH / "sampled"
    VALUES_PATH = OUTPUT_PATH / "values"
    RESULT_PATH = OUTPUT_PATH / "results"
    PLOT_PATH = OUTPUT_PATH / "plots"

    @staticmethod
    def valuation_results(
        experiment_names: OneOrMany[str],
        model_names: OneOrMany[str],
        dataset_names: OneOrMany[str],
        method_names: OneOrMany[str],
        repetition_ids: OneOrMany[int],
    ) -> pd.DataFrame:
        experiment_names = ensure_list(experiment_names)
        model_names = ensure_list(model_names)
        dataset_names = ensure_list(dataset_names)
        method_names = ensure_list(method_names)
        repetition_ids = ensure_list(repetition_ids)

        rows = []
        for (
            experiment_name,
            model_name,
            dataset_name,
            method_name,
            repetition_id,
        ) in tqdm(
            list(
                product(
                    experiment_names,
                    model_names,
                    dataset_names,
                    method_names,
                    repetition_ids,
                )
            ),
            desc="Loading valuation results...",
            ncols=120,
        ):
            base_path = (
                Accessor.VALUES_PATH
                / experiment_name
                / model_name
                / dataset_name
                / str(repetition_id)
            )
            with open(base_path / f"valuation.{method_name}.pkl", "rb") as f:
                valuation = pickle.load(f)
            with open(base_path / f"valuation.{method_name}.stats.json", "r") as f:
                stats = json.load(f)

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "method_name": method_name,
                    "repetition_id": repetition_id,
                    "valuation": valuation,
                }
                | stats
            )

        return pd.DataFrame(rows)

    @staticmethod
    def metrics_and_curves(
        experiment_names: OneOrMany[str],
        model_names: OneOrMany[str],
        dataset_names: OneOrMany[str],
        method_names: OneOrMany[str],
        repetition_ids: OneOrMany[int],
        metric_names: OneOrMany[str],
    ) -> pd.DataFrame:
        experiment_names = ensure_list(experiment_names)
        model_names = ensure_list(model_names)
        dataset_names = ensure_list(dataset_names)
        method_names = ensure_list(method_names)
        repetition_ids = ensure_list(repetition_ids)

        rows = []
        for (
            experiment_name,
            model_name,
            dataset_name,
            method_name,
            repetition_id,
        ) in tqdm(
            list(
                product(
                    experiment_names,
                    model_names,
                    dataset_names,
                    method_names,
                    repetition_ids,
                )
            ),
            desc="Loading metrics and curves...",
            ncols=120,
        ):
            base_path = (
                Accessor.RESULT_PATH
                / experiment_name
                / model_name
                / dataset_name
                / str(repetition_id)
                / method_name
            )
            for metric_name in metric_names:
                metric = pd.read_csv(base_path / f"{metric_name}.csv")
                metric = metric.iloc[-1, -1]

                curve = pd.read_csv(base_path / f"{metric_name}.curve.csv")
                curve.index = curve[curve.columns[0]]
                curve = curve.drop(columns=[curve.columns[0]]).iloc[:, -1]

                rows.append(
                    {
                        "experiment_name": experiment_name,
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "method_name": method_name,
                        "repetition_id": repetition_id,
                        "metric_name": metric_name,
                        "metric": metric,
                        "curve": curve,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def datasets(
        experiment_names: OneOrMany[str],
        dataset_names: OneOrMany[str],
        repetition_ids: OneOrMany[int],
    ) -> pd.DataFrame:
        experiment_names = ensure_list(experiment_names)
        dataset_names = ensure_list(dataset_names)
        repetition_ids = ensure_list(repetition_ids)

        rows = []
        for (
            experiment_name,
            dataset_name,
            repetition_id,
        ) in tqdm(
            list(
                product(
                    experiment_names,
                    dataset_names,
                    repetition_ids,
                )
            ),
            desc="Loading datasets...",
            ncols=120,
        ):
            base_path = (
                Accessor.SAMPLED_PATH
                / experiment_name
                / dataset_name
                / str(repetition_id)
            )
            with open(base_path / f"val_set.pkl", "rb") as file:
                val_set = pickle.load(file)

            with open(base_path / f"test_set.pkl", "rb") as file:
                test_set = pickle.load(file)

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                    "repetition_id": repetition_id,
                    "val_set": val_set,
                    "test_set": test_set,
                }
            )

        return pd.DataFrame(rows)
