from copy import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pydvl.utils import Dataset, Scorer, Utility
from pydvl.value.shapley.classwise import CSScorer
from sklearn.metrics import roc_auc_score

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.utils import instantiate_model, setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


@dataclass
class ExperimentTwoResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame


def run_experiment_two(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_methods: Dict[str, Dict],
    perc_flip_labels: float = 0.2,
) -> ExperimentTwoResult:
    logger.debug("Noise flip example")
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_methods.keys())
    )
    result = ExperimentTwoResult(
        metric=copy(base_frame), valuation_results=copy(base_frame)
    )

    for dataset_idx, (val_dataset, test_dataset) in datasets.items():
        scorer = CSScorer()
        model = instantiate_model(model_name)

        logger.info("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)

        # flip the train labels
        num_data_indices = int(perc_flip_labels * len(val_dataset.y_train))
        p = np.random.permutation(len(val_dataset.y_train))[:num_data_indices]
        val_dataset.y_train[p] = 1 - val_dataset.y_train[p]
        test_dataset.y_train = val_dataset.y_train

        for valuation_method_name, valuation_method_kwargs in valuation_methods.items():
            logger.info(f"{valuation_method_name=}")
            logger.info(f"Computing values using '{valuation_method_name}'.")

            values = compute_values(
                utility,
                valuation_method=valuation_method_name,
                **valuation_method_kwargs,
            )
            result.valuation_results.loc[dataset_idx, valuation_method_name] = values

            p = np.argsort(values)[:num_data_indices]
            val_dataset.y_train[p] = 1 - val_dataset.y_train[p]
            test_dataset.y_train = val_dataset.y_train

            logger.info(
                "Computing best data points removal score on separate test set."
            )

            y_true = test_dataset.y_test
            y_pred = utility.model.predict(test_dataset.x_test)
            auc = roc_auc_score(y_true, y_pred)
            result.metric.loc[dataset_idx, valuation_method_name] = auc

    return result
