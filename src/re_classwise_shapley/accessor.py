import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

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

    def __init__(self, experiment_name: str, model_name: str):
        self._experiment_name = experiment_name
        self.__model_name = model_name

    def valuation_results(
        self,
        dataset_names: OneOrMany[str],
        method_names: OneOrMany[str],
        repetition_ids: OneOrMany[int],
    ) -> Dict[str, Dict[str, NDArray[np.float_]]]:
        """
        Fetches the valuation results for a given dataset, method and repetition.
        Args:
            dataset_names: List of dataset_names to load.
            method_names: List of method_names to load.
            repetition_ids: List of repetition_ids to load.

        Returns:
            A three-dimensional array of valuation results. The first dimension is the
            dataset, the second dimension is the method and the third dimension is the
            repetition (in numerical order).
        """
        dataset_names = ensure_list(dataset_names)
        method_names = ensure_list(method_names)
        repetition_ids = ensure_list(repetition_ids)

        valuation_results = {}
        for dataset_name in dataset_names:
            valuation_results[dataset_name] = {}
            for method_name in method_names:
                valuation_result_components = []
                for repetition_id in repetition_ids:
                    valuation_results_path = (
                        Accessor.VALUES_PATH
                        / self._experiment_name
                        / self.__model_name
                        / dataset_name
                        / str(repetition_id)
                    )
                    with open(
                        valuation_results_path / f"valuation.{method_name}.pkl", "rb"
                    ) as file:
                        valuation_result_components.append(pickle.load(file).values)

                valuation_results[dataset_name][method_name] = np.stack(
                    valuation_result_components
                )

        return valuation_results
