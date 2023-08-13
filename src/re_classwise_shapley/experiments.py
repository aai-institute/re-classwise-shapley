import pickle
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility
from pydvl.value.result import ValuationResult

from re_classwise_shapley.eval.metric import (
    pr_curve_ranking,
    weighted_reciprocal_diff_average,
)
from re_classwise_shapley.types import ValuationMethodsFactory
from re_classwise_shapley.utils import setup_logger

logger = setup_logger()


@dataclass
class ExperimentResult:
    valuation_results: pd.Series
    metric: pd.Series
    curves: Optional[pd.Series]
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
    val_test_set: Tuple[Dataset, Dataset],
    valuation_methods: ValuationMethodsFactory,
    *,
    data_pre_process_fn: Callable[[NDArray[int]], Tuple[NDArray[int], Dict]] = None,
    metric_functions: Dict[
        str,
        Callable[[Utility, ValuationResult, Dict], Tuple[float, Optional[pd.Series]]],
    ],
) -> ExperimentResult:
    base_series = pd.Series(index=list(valuation_methods.keys()))
    val_dataset, test_dataset = val_test_set
    result = ExperimentResult(
        metric=copy(base_series),
        valuation_results=copy(base_series),
        curves=None,
        val_set=val_dataset,
        test_set=test_dataset,
    )
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
        result.valuation_results.loc[valuation_method_name] = values

        if values is None:
            result.metric.loc[valuation_method_name] = values
            continue

        logger.info("Computing best data points removal score on separate test set.")
        test_utility = Utility(
            data=test_dataset, model=model, scorer=Scorer(scoring="accuracy")
        )

        metrics = {}
        curves = {}
        for metric_name, metric_fn in metric_functions.items():
            metric, graph = metric_fn(test_utility, values, info)
            metrics[metric_name] = metric
            curves[metric_name] = graph

        result.metric.loc[valuation_method_name] = [metrics]

        if len(curves) > 0 and any([v is not None for v in curves]):
            if result.curves is None:
                result.curves = copy(base_series)

            result.curves.loc[valuation_method_name] = [curves]

    if result.metric is not None:
        result.metric = result.metric.apply(lambda x: x[0])

    if result.curves is not None:
        result.curves = result.curves.apply(lambda x: x[0])

    return result


def experiment_wad(
    model: SupervisedModel,
    val_test_set: Tuple[Dataset, Dataset],
    valuation_methods_factory: ValuationMethodsFactory,
    test_model: SupervisedModel = None,
    progress: bool = False,
) -> ExperimentResult:
    """
    Runs an experiment using the weighted accuracy drop (WAD) introduced in [1]. This function can be reused for the
    first and third experiments inside the paper.

    :param model: Model which shall be used for evaluation.
    :param val_test_set: Tuple of validation and test set.
    :param valuation_methods_factory: All valuation methods to be used.
    :param test_model: The current test model which shall be used.
    :param progress: Whether to display a progress bar.
    :return: An ExperimentResult object with the gathered data.
    """

    def _weighted_accuracy_drop(
        test_utility: Utility,
        values: ValuationResult,
        info: Dict,
        highest_first: bool = True,
    ) -> Tuple[float, pd.Series]:
        eval_utility = Utility(
            data=test_utility.data,
            model=test_model if test_model is not None else model,
            scorer=Scorer("accuracy", default=0),
        )
        weighted_accuracy_drop, graph = weighted_reciprocal_diff_average(
            u=eval_utility,
            values=values,
            progress=progress,
            highest_first=highest_first,
        )
        return float(weighted_accuracy_drop), graph

    result = _dispatch_experiment(
        model,
        val_test_set,
        valuation_methods_factory,
        data_pre_process_fn=None,
        metric_functions={  # type: ignore
            "highest_wad_drop": partial(_weighted_accuracy_drop, highest_first=True),
            "lowest_wad_drop": partial(_weighted_accuracy_drop, highest_first=False),
        },
    )
    result.metric_name = "wad"
    return result


def experiment_noise_removal(
    model: SupervisedModel,
    val_test_set: Tuple[Dataset, Dataset],
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
        ranked_list = list(np.argsort(values))
        ranked_list = test_utility.data.indices[ranked_list]
        recall, precision, score = pr_curve_ranking(info["idx"], ranked_list)
        logger.debug("Computing precision-recall curve on separate test set..")
        graph = pd.Series(precision, index=recall)
        graph = graph[~graph.index.duplicated(keep="first")]
        graph = graph.sort_index(ascending=True)
        return score, graph

    result = _dispatch_experiment(
        model,
        val_test_set,
        valuation_methods_factory,
        data_pre_process_fn=_flip_labels,
        metric_functions={"roc_precision_recall": _roc_auc},
    )
    result.metric_name = "roc_auc"
    return result
