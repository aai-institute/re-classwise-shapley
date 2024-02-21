import pickle

from sklearn.metrics import precision_recall_curve, roc_auc_score

from csshapley22.log import setup_logger

setup_logger()

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pydvl.utils import ClasswiseScorer, Dataset, Scorer, SupervisedModel, Utility
from pydvl.value.result import ValuationResult

from csshapley22.metrics.weighted_reciprocal_average import (
    weighted_reciprocal_diff_average,
)
from csshapley22.types import ValTestSetFactory, ValuationMethodsFactory
from csshapley22.utils import setup_logger, timeout

logger = setup_logger()


@dataclass
class ExperimentResult:
    valuation_results: pd.DataFrame
    metric: pd.DataFrame
    curves: Optional[pd.DataFrame]
    val_set: Dataset
    test_set: Dataset
    metric_name: str = None

    def store(self, output_dir: Path) -> "ExperimentResult":
        logger.info("Saving results to disk")
        output_dir.mkdir(parents=True, exist_ok=True)

        validation_set_path = str(output_dir / "val_set.pkl")
        test_set_path = str(output_dir / "test_set.pkl")
        for set_path, set in [
            (validation_set_path, self.val_set),
            (test_set_path, self.test_set),
        ]:
            with open(set_path, "wb") as file:
                pickle.dump(set, file)

        self.metric.to_csv(output_dir / "metric.csv")
        self.valuation_results.to_pickle(output_dir / "valuation_results.pkl")
        if self.curves is not None:
            self.curves.to_pickle(output_dir / "curves.pkl")

        return self


def _dispatch_experiment(
    model: SupervisedModel,
    datasets: ValTestSetFactory,
    valuation_methods: ValuationMethodsFactory,
    *,
    data_pre_process_fn: Callable[[NDArray[int]], Tuple[NDArray[int], Dict]] = None,
    metric_fn: Callable[
        [Utility, ValuationResult, Dict], Tuple[float, Optional[pd.Series]]
    ],
) -> ExperimentResult:
    base_frame = pd.DataFrame(
        index=list(datasets.keys()), columns=list(valuation_methods.keys())
    )
    dataset_idxs = list(datasets.keys())
    assert len(dataset_idxs) == 1, "Only one dataset is supported for now."
    dataset_idx = dataset_idxs[0]
    dataset_factory = datasets[dataset_idx]
    val_dataset, test_dataset = dataset_factory()
    result = ExperimentResult(
        metric=copy(base_frame),
        valuation_results=copy(base_frame),
        curves=None,
        val_set=val_dataset,
        test_set=test_dataset,
    )
    logger.info(f"Loading dataset '{dataset_idx}'.")
    logger.debug("Creating utility")  # type: ignore

    info = None
    if data_pre_process_fn is not None:
        val_dataset.y_train, info = data_pre_process_fn(val_dataset.y_train)
        test_dataset.y_train = val_dataset.y_train

    for valuation_method_name, valuation_method in valuation_methods.items():
        scorer = Scorer("accuracy", default=0.0)
        utility = Utility(data=val_dataset, model=model, scorer=scorer)
        logger.info(f"Computing values using '{valuation_method_name}'.")

        # valuation_method = timeout(10800)(valuation_method)
        values = valuation_method(utility)
        result.valuation_results.loc[dataset_idx, valuation_method_name] = values

        if values is None:
            result.metric.loc[dataset_idx, valuation_method_name] = values
            continue

        logger.info("Computing best data points removal score on separate test set.")
        test_utility = Utility(
            data=test_dataset, model=model, scorer=Scorer(scoring="accuracy")
        )
        metric, graph = metric_fn(test_utility, values, info)
        result.metric.loc[dataset_idx, valuation_method_name] = metric

        if graph is not None:
            if result.curves is None:
                result.curves = copy(base_frame)

            result.curves.loc[dataset_idx, valuation_method_name] = [graph]

    if result.curves is not None:
        result.curves = result.curves.applymap(lambda x: x[0])

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
        test_utility: Utility, values: ValuationResult, info: Dict
    ) -> Tuple[float, pd.Series]:
        eval_utility = Utility(
            data=test_utility.data,
            model=test_model if test_model is not None else model,
            scorer=Scorer("accuracy", default=0),
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
    def _flip_labels(labels: NDArray[int]) -> Tuple[NDArray[int], Dict]:
        num_data_indices = int(perc_flip_labels * len(labels))
        p = np.random.permutation(len(labels))[:num_data_indices]
        labels[p] = 1 - labels[p]
        return labels, {"idx": p, "num_flipped": num_data_indices}

    def _roc_auc(
        test_utility: Utility, values: ValuationResult, info: Dict
    ) -> Tuple[float, pd.Series]:
        y_true = np.zeros(len(test_utility.data.y_train), dtype=int)
        y_true[info["idx"]] = 1

        y_pred = np.zeros(len(test_utility.data.y_train), dtype=int)
        y_pred[np.argsort(values)[: info["num_flipped"]]] = 1

        logger.debug("Computing precision-recall curve on separate test set..")
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            y_pred.astype(float),
        )
        graph = pd.Series(precision, index=recall)
        graph = graph[~graph.index.duplicated(keep="first")]
        graph = graph.sort_index(ascending=True)
        return roc_auc_score(y_true, y_pred), graph

    result = _dispatch_experiment(
        model,
        datasets,
        valuation_methods_factory,
        data_pre_process_fn=_flip_labels,
        metric_fn=_roc_auc,
    )
    result.metric_name = "roc_auc"
    return result
