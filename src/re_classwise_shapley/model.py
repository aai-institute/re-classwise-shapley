from typing import Dict

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import SupervisedModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from re_classwise_shapley.types import Seed


def instantiate_model(
    model_name: str, model_kwargs: Dict, seed: Seed = None
) -> SupervisedModel:
    seed = seed.generate_state(1)[0]
    random_state = np.random.RandomState(seed)

    if model_name == "gradient_boosting_classifier":
        model = make_pipeline(
            GradientBoostingClassifier(**model_kwargs, random_state=random_state)
        )
    elif model_name == "logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**model_kwargs, random_state=random_state),
        )
    elif model_name == "knn":
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(**model_kwargs, random_state=random_state),
        )
    elif model_name == "svm":
        model = make_pipeline(
            StandardScaler(), SVC(**model_kwargs, random_state=random_state)
        )
    elif model_name == "mlp":
        model = make_pipeline(
            StandardScaler(), MLPClassifier(**model_kwargs, random_state=random_state)
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return SingleClassProxySupervisedModel(model)


class SingleClassProxySupervisedModel(SupervisedModel):
    def __init__(self, model: SupervisedModel):
        self.model = model
        self._unique_cls = None

    def fit(self, x: NDArray[np.float_], y: NDArray[np.int_]):
        if len(np.unique(y)) == 1:
            self._unique_cls = y[0]
        else:
            self._unique_cls = None
            self.model.fit(x, y)

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.int_]:
        if self._unique_cls is None:
            return self.model.predict(x)
        else:
            return np.ones(len(x), dtype=int) * self._unique_cls

    def score(self, x: NDArray[np.float_], y: NDArray[np.int_]) -> float:
        return self.model.score(x, y)
