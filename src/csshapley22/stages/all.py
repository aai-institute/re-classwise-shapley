import logging

from csshapley22.log import setup_logger

setup_logger()

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pydvl.utils import ClassWiseScorer, Dataset, Scorer, SupervisedModel, Utility
from pydvl.value.result import ValuationResult

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.types import ValTestSetFactory, ValuationMethodsFactory
from csshapley22.utils import setup_logger, timeout
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


@dataclass
class ExperimentResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame
    graphs: Optional[pd.DataFrame]
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
    data_pre_process_fn: Callable[[NDArray[int]], NDArray[int]] = None,
    metric_fn: Callable[[Utility, ValuationResult], Tuple[float, Optional[pd.Series]]],
) -> ExperimentResult:
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_methods.keys())
    )
    result = ExperimentResult(
        metric=copy(base_frame), valuation_results=copy(base_frame), graphs=None
    )

    for dataset_idx, (val_dataset, test_dataset) in datasets.items():
        logger.info(f"Loading dataset '{dataset_idx}'.")
        scorer = ClassWiseScorer("accuracy")

        logger.debug("Creating utility")
        utility = Utility(data=val_dataset, model=model, scorer=scorer)  # type: ignore

        if data_pre_process_fn is not None:
            val_dataset.y_train = data_pre_process_fn(val_dataset.y_train)
            test_dataset.y_train = val_dataset.y_train

        for valuation_method_name, valuation_method in valuation_methods.items():
            logger.info(f"Computing values using '{valuation_method_name}'.")

            # valuation_method = timeout(1800)(valuation_method)
            values = valuation_method(utility)
            result.valuation_results.loc[dataset_idx, valuation_method_name] = values

            if values is None:
                result.metric.loc[dataset_idx, valuation_method_name] = values
                continue

            logger.info(
                "Computing best data points removal score on separate test set."
            )
            test_utility = Utility(
                data=test_dataset, model=model, scorer=Scorer(scoring="accuracy")
            )
            metric, graph = metric_fn(test_utility, values)
            result.metric.loc[dataset_idx, valuation_method_name] = metric

            if graph is not None:
                if result.graphs is None:
                    result.graphs = copy(base_frame)

                result.graphs.loc[dataset_idx, valuation_method_name] = [graph]

    if result.graphs is not None:
        result.graphs = result.graphs.applymap(lambda x: x[0])

    return result


def experiment_wad(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods_factory: ValuationMethodsFactory,
    test_model: SupervisedModel = None,
    progress: bool = False,
) -> ExperimentResult:
    """
    Runs an experiment using the weighted accuracy drop (WAD) introduced in [1]. This function can be reused for the
    first and third experiments inside the paper.

    :param model: Model which shall be used for evaluation.
    :param datasets: A dictionary containing validation and test set tuples
    :param valuation_methods_factory: All valuation methods to be used.
    :param test_model: The current test model which shall be used.
    :param progress: Whether to display a progress bar.
    :return: An ExperimentResult object with the gathered data.
    """

    def _weighted_accuracy_drop(
        test_utility: Utility, values: ValuationResult
    ) -> Tuple[float, pd.Series]:
        eval_utility = Utility(
            data=test_utility.data,
            model=test_model if test_model is not None else model,
            scorer=Scorer("accuracy"),
        )
        weighted_accuracy_drop, graph = weighted_reciprocal_diff_average(
            u=eval_utility, values=values, progress=progress
        )
        return float(weighted_accuracy_drop), graph

    result = _dispatch_experiment(
        model,
        datasets,
        valuation_methods_factory,
        data_pre_process_fn=None,
        metric_fn=_weighted_accuracy_drop,
    )
    result.metric_name = "wad"
    return result


def experiment_noise_removal(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods_factory: ValuationMethodsFactory,
    perc_flip_labels: float = 0.2,
) -> ExperimentResult:
    def _flip_labels(labels: NDArray[int]) -> NDArray[int]:
        num_data_indices = int(perc_flip_labels * len(labels))
        p = np.random.permutation(len(labels))[:num_data_indices]
        labels[p] = 1 - labels[p]
        return labels

    def _roc_auc(test_utility: Utility, values: ValuationResult) -> Tuple[float, None]:
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
        return u(u.data.indices), None

    result = _dispatch_experiment(
        model,
        datasets,
        valuation_methods_factory,
        data_pre_process_fn=_flip_labels,
        metric_fn=_roc_auc,
    )
    result.metric_name = "roc_auc"
    return result
