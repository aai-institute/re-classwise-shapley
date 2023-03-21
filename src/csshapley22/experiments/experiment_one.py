from copy import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd
from pydvl.utils import Dataset, Scorer, Utility
from pydvl.value import ValuationResult
from pydvl.value.shapley.classwise import CSScorer

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.utils import instantiate_model, setup_logger

logger = setup_logger()


@dataclass
class ExperimentOneResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame


def run_experiment_one(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_functions: Dict[str, List[Callable[[Utility], ValuationResult]]],
) -> ExperimentOneResult:
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_functions.keys())
    )
    result = ExperimentOneResult(
        metric=copy(base_frame), valuation_results=copy(base_frame)
    )

    for dataset_idx, (val_dataset, test_dataset) in datasets.items():
        scorer = CSScorer()
        model = instantiate_model(model_name)

        logger.info("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)

        for valuation_method_idx, valuation_method in valuation_functions.items():
            logger.info(f"{valuation_method=}")
            logger.info("Computing values")

            values = valuation_method(utility)
            result.valuation_results.loc[dataset_idx, valuation_method_idx] = values

            logger.info("Computing best data points removal score")
            accuracy_utility = Utility(
                data=val_dataset, model=model, scorer=Scorer(scoring="accuracy")
            )
            weighted_accuracy_drop = weighted_reciprocal_diff_average(
                u=accuracy_utility, values=values, progress=True
            )
            result.metric.loc[
                dataset_idx, valuation_method_idx
            ] = weighted_accuracy_drop

    return result
