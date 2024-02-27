from typing import Any, List, Protocol

import numpy as np
import pandas as pd

from re_classwise_shapley.io import Accessor


class FunctionalRequest(Protocol):
    def get(self, *args, **kwargs) -> Any:
        pass


class FunctionalCurveRequest(FunctionalRequest):
    """
    Concept to request information from another curve. This is a specialization of
    a more general concept called FunctionalRequests.
    """

    def __init__(self, arg_name: str, method_name: str, repetitions_ids: List[int]):
        self.arg_name = arg_name
        self.method_name = method_name
        self.repetitions_ids = repetitions_ids

    def request(
        self,
        experiment_name: str,
        model_name: str,
        dataset_name: str,
        curve_name: str,
    ) -> pd.Series:
        all = list(
            Accessor.curves(
                experiment_name,
                model_name,
                dataset_name,
                self.method_name,
                curve_name,
                self.repetitions_ids,
            )["curve"]
        )
        return pd.Series(np.stack(all, axis=0).mean(axis=0), index=all[0].index)
