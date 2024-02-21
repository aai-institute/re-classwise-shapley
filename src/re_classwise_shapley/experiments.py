import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from dvc.api import params_show
from numpy.random import SeedSequence
from numpy.typing import NDArray
from pydvl.utils import Dataset, Scorer, SupervisedModel, Utility
from pydvl.value.result import ValuationResult

from re_classwise_shapley.config import Config
from re_classwise_shapley.data import fetch_and_sample_val_test_dataset
from re_classwise_shapley.io import parse_valuation_method_dict
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.model import instantiate_model
from re_classwise_shapley.types import Seed, ValuationMethodDict
from re_classwise_shapley.utils import clear_folder, init_random_seed

logger = setup_logger(__name__)


@dataclass
class ExperimentResult:
    valuation_results: Dict[str, ValuationResult]
    metric: Dict[str, Dict[str, float]]
    curves: Dict[str, Dict[str, pd.Series]]
    val_set: Dataset
    test_set: Dataset

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

        metrics = pd.DataFrame(self.metric)
        metrics.index.name = "metrics"
        metrics.to_csv(output_dir / "metrics.csv")

        if self.curves is not None:
            for valuation_method_name, valuation_method_curves in self.curves.items():
                for metric_name, curve in valuation_method_curves.items():
                    path_name = f"curve_{valuation_method_name}_{metric_name}.csv"
                    curve.to_csv(output_dir / path_name)

        return self


def run_experiment(
    model: SupervisedModel,
    val_test_set: Tuple[Dataset, Dataset],
    valuation_methods: ValuationMethodDict,
    *,
    label_preprocessor: Callable[[NDArray[int]], Tuple[NDArray[int], Dict]] = None,
    metrics: Dict[
        str,
        Callable[[Utility, ValuationResult, Dict], Tuple[float, Optional[pd.Series]]],
    ] = None,
    seed: Seed = None,
) -> ExperimentResult:
    val_dataset, test_dataset = val_test_set
    result = ExperimentResult(
        valuation_results=dict(),
        metric=dict(),
        curves=dict(),
        val_set=val_dataset,
        test_set=test_dataset,
    )
    logger.debug("Creating utility")  # type: ignore

    info = None
    if label_preprocessor is not None:
        val_dataset.y_train, info = label_preprocessor(val_dataset.y_train)
        test_dataset.y_train = val_dataset.y_train

    valuation_method_names = list(valuation_methods.keys())
    seeds = seed.spawn(len(valuation_method_names))

    for idx, valuation_method_name in enumerate(valuation_method_names):
        valuation_method = valuation_methods[valuation_method_name]
        logger.info(f"Computing values using '{valuation_method_name}'.")
        values = valuation_method(
            Utility(
                data=val_dataset,
                model=deepcopy(model),
                scorer=Scorer("accuracy", default=0.0),
            ),
            seed=seeds[idx],
        )
        result.valuation_results[valuation_method_name] = values
        test_utility = Utility(
            data=test_dataset, model=deepcopy(model), scorer=Scorer(scoring="accuracy")
        )

        for metric_name, metric_fn in metrics.items():
            logger.info(f"Computing {metric_name=}.")
            metric, graph = metric_fn(test_utility, values, info)
            if valuation_method_name not in result.metric:
                result.metric[valuation_method_name] = dict()
                result.curves[valuation_method_name] = dict()

            result.metric[valuation_method_name][metric_name] = metric
            result.curves[valuation_method_name][metric_name] = graph

    return result


def run_and_store_experiment(
    experiment_name: str,
    output_dir: Path,
    loader_kwargs: Callable[[Optional[SeedSequence]], Dict[str, Any]],
    dataset_name: str,
    model_name: str,
    n_repetitions: int = 1,
    seed: int = None,
):
    logger.info(Config.DOUBLE_BREAK)
    logger.info(f"Start {experiment_name=} with '{model_name=}' on '{dataset_name=}.")
    logger.info(f"Args: \t{seed=}, \n\t\t{output_dir=}.")

    seed_sequence = init_random_seed(seed)
    seeds = seed_sequence.spawn(n_repetitions)

    params = params_show()
    valuation_methods = parse_valuation_method_dict(
        params["valuation_methods"],
        active_valuation_methods=params["active"]["valuation_methods"],
        global_kwargs=params["settings"]["parallel"],
    )

    for repetition in range(n_repetitions):
        logger.info(Config.SINGLE_BREAK)
        logger.info(f"Executing {repetition=}.")

        model_kwargs = params["models"][model_name]
        sub_seeds = seeds[repetition].spawn(4)
        model = instantiate_model(model_name, model_kwargs, seed=sub_seeds[0])

        val_test_set_kwargs = params["datasets"][dataset_name]
        val_test_set = fetch_and_sample_val_test_dataset(
            dataset_name, val_test_set_kwargs["split"], seed=sub_seeds[1]
        )

        experiment_kwargs = loader_kwargs(sub_seeds[2])
        repetition_output_dir = output_dir / f"{repetition=}"

        try:
            result = run_experiment(
                model=model,
                val_test_set=val_test_set,
                valuation_methods=valuation_methods,
                **experiment_kwargs,
                seed=sub_seeds[3],
            )
            result.store(repetition_output_dir)

        except Exception as e:
            clear_folder(repetition_output_dir)
            logger.error(
                f"Error while executing experiment: {e}. Skipping experiment and"
                f" continue with next one."
            )
            raise e
