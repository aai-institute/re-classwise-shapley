from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, TypeVar

import numpy as np
import pandas as pd
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility
from pydvl.value import ValuationResult
from pydvl.value.shapley.classwise import CSScorer

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.types import ValTestSetFactory, ValuationMethodsFactory
from csshapley22.utils import setup_logger
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


@dataclass
class ExperimentResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame
    metric_name: str = None

    def store(self, output_dir: Path) -> "ExperimentResult":
        logger.info("Saving results to disk")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.metric.to_csv(output_dir / "metric.csv")
        self.valuation_results.to_csv(output_dir / "valuation_results.csv")
        return self


def _dispatch_experiment(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods: ValuationMethodsFactory,
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
        # Create a mock scorer
        scorer = CSScorer()

        logger.info("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)  # type: ignore

        if data_pre_process_fn is not None:
            val_dataset.y_train = data_pre_process_fn(val_dataset.y_train)
            test_dataset.y_train = val_dataset.y_train

        for valuation_method_name, valuation_method in valuation_methods.items():
            logger.info(f"{valuation_method_name=}")
            logger.info(f"Computing values using '{valuation_method_name}'.")

            values = valuation_method(utility)
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


def experiment_wad(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods: ValuationMethodsFactory,
    test_model: SupervisedModel = None,
) -> ExperimentResult:
    """
    Runs an experiment using the weighted accuracy drop (WAD) introduced in [1]. This function can be reused for the
    first and third experiments inside the paper.

    :param model: Model which shall be used for evaluation.
    :param datasets: A dictionary containing validation and test set tuples
    :param valuation_methods: All valuation methods to be used.
    :param test_model: The current test model which shall be used.
    :return: An ExperimentResult object with the gathered data.
    """

    def _weighted_accuracy_drop(
        test_utility: Utility, values: ValuationResult
    ) -> float:
        eval_utility = Utility(
            data=test_utility.data,
            model=test_model,
            scorer=Scorer(scoring="accuracy"),
        )
        weighted_accuracy_drop = weighted_reciprocal_diff_average(
            u=eval_utility, values=values, progress=True
        )
        return float(weighted_accuracy_drop)

    result = _dispatch_experiment(
        model,
        datasets,
        valuation_methods,
        data_pre_process_fn=None,
        metric_fn=_weighted_accuracy_drop,
    )
    result.metric_name = "wad"
    return result


def experiment_noise_removal(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods: ValuationMethodsFactory,
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

    result = _dispatch_experiment(
        model,
        datasets,
        valuation_methods,
        data_pre_process_fn=_flip_labels,
        metric_fn=_roc_auc,
    )
    result.metric_name = "roc_auc"
    return result
