from copy import copy
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from pydvl.utils import Dataset, Scorer, Utility
from pydvl.value import ValuationResult
from pydvl.value.shapley.classwise import CSScorer

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.utils import instantiate_model, setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


@dataclass
class ExperimentResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame
    metric_name: str = None


def run_experiment(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_methods: Dict[str, Dict],
    *,
    data_pre_process_fn: Callable[[pd.Series], pd.Series] = None,
    metric_fn: Callable[[Utility, ValuationResult], float],
) -> ExperimentResult:
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_methods.keys())
    )
    result = ExperimentResult(
        metric=copy(base_frame), valuation_results=copy(base_frame)
    )

    for dataset_idx, (val_dataset, test_dataset) in datasets.items():
        scorer = CSScorer()
        model = instantiate_model(model_name)

        logger.info("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)

        if data_pre_process_fn is not None:
            val_dataset.y_train = data_pre_process_fn(val_dataset.y_train)
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

            logger.info(
                "Computing best data points removal score on separate test set."
            )
            test_utility = Utility(
                data=test_dataset, model=model, scorer=Scorer(scoring="accuracy")
            )
            result.metric.loc[dataset_idx, valuation_method_name] = metric_fn(
                test_utility, values
            )

    return result


def run_experiment_one(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_methods: Dict[str, Dict],
) -> ExperimentResult:
    def _weighted_accuracy_drop(
        test_utility: Utility, values: ValuationResult
    ) -> float:
        eval_utility = Utility(
            data=test_utility.data,
            model=test_utility.model,
            scorer=Scorer(scoring="accuracy"),
        )
        weighted_accuracy_drop = weighted_reciprocal_diff_average(
            u=eval_utility, values=values, progress=True
        )
        return float(weighted_accuracy_drop)

    result = run_experiment(
        model_name,
        datasets,
        valuation_methods,
        data_pre_process_fn=None,
        metric_fn=_weighted_accuracy_drop,
    )
    result.metric_name = "wad"
    return result


def run_experiment_two(
    model_name: str,
    datasets: Dict[str, Tuple[Dataset, Dataset]],
    valuation_methods: Dict[str, Dict],
    perc_flip_labels: float = 0.2,
) -> ExperimentResult:
    def _flip_labels(labels: pd.Series) -> pd.Series:
        num_data_indices = int(perc_flip_labels * len(labels))
        p = np.random.permutation(len(labels))[:num_data_indices]
        labels.iloc[p] = 1 - labels.iloc[p]
        return labels

    def _roc_auc(test_utility: Utility, values: ValuationResult) -> float:
        n = len(test_utility.data.y_train)
        num_data_indices = int(perc_flip_labels * n)
        p = np.argsort(values)[:num_data_indices]
        test_utility.data.y_train[p] = 1 - test_utility.data.y_train[p]

        logger.debug("Computing best data points removal score on separate test set.")
        u = Utility(
            data=test_utility.data,
            model=test_utility.model,
            scorer=Scorer(scoring="roc_auc"),
        )
        return u(u.data.indices)

    result = run_experiment(
        model_name,
        datasets,
        valuation_methods,
        data_pre_process_fn=_flip_labels,
        metric_fn=_roc_auc,
    )
    result.metric_name = "roc_auc"
    return result
