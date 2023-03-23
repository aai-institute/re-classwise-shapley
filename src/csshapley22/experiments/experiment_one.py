from copy import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from pydvl.utils import Dataset, Scorer, Utility
from pydvl.value.shapley.classwise import CSScorer

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.utils import instantiate_model, setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


@dataclass
class ExperimentOneResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame


def run_experiment_one(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_methods: Dict[str, Dict],
) -> ExperimentOneResult:
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_methods.keys())
    )
    result = ExperimentOneResult(
        metric=copy(base_frame), valuation_results=copy(base_frame)
    )

    for dataset_idx, (val_dataset, test_dataset) in datasets.items():
        scorer = CSScorer()
        model = instantiate_model(model_name)

        logger.info("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)

        for valuation_method_name, valuation_method_kwargs in valuation_methods.items():
            logger.info(f"{valuation_method_name=}")
            logger.info(f"Computing values using '{valuation_method_name}'.")

            values = compute_values(
                utility,
                valuation_method=valuation_method_name,
                **valuation_method_kwargs,
            )
            result.valuation_results.loc[dataset_idx, valuation_method_name] = values

            logger.info(
                "Computing best data points removal score on separate test set."
            )
            accuracy_utility = Utility(
                data=test_dataset, model=model, scorer=Scorer(scoring="accuracy")
            )
            weighted_accuracy_drop = weighted_reciprocal_diff_average(
                u=accuracy_utility, values=values, progress=True
            )
            result.metric.loc[
                dataset_idx, valuation_method_name
            ] = weighted_accuracy_drop

    return result
