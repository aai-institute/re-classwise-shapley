from typing import Dict, Optional

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


def instantiate_model(
    model_name: str, model_kwargs: Dict, seed: Optional[int] = None
) -> SupervisedModel:
    """
    Instantiates a model with the given name and keyword arguments. The model is wrapped
    in a `SingleClassProxySupervisedModel`. All models use standard scaling except for
    gradient boosting classifiers.

    Args:
        model_name: Name of the model to instantiate. The name must be one of the
            following: `gradient_boosting_classifier`, `logistic_regression`, `knn`,
            `svm`, `mlp`.
        model_kwargs: Keyword arguments to pass to the model constructor.
        seed: Seed to use for the model. If `None`, no seed is used.

    Returns:
        The instantiated model.
    """
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
            KNeighborsClassifier(**model_kwargs),
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
    """
    A proxy model that returns a single class if the training data only contains a single
    class. This is necessary for the valuation methods that require a binary classifier.

    Args:
        model: The model to wrap.
    """

    def __init__(self, model: SupervisedModel):
        self.model = model
        self._unique_cls = None

    def fit(self, features: NDArray[np.float_], labels: NDArray[np.int_]):
        """
        Fits the model to the given data. If the data only contains a single class, the
        model is not fitted and the class is stored.

        Args:
            features: Features to fit the model on.
            labels: Labels to fit the model on.
        """
        if len(np.unique(labels)) == 1:
            self._unique_cls = labels[0]
        else:
            self._unique_cls = None
            self.model.fit(features, labels)

    def predict(self, features: NDArray[np.float_]) -> NDArray[np.int_]:
        """
        Predicts the class of the given data. If the data only contains a single class,
        the class is returned for all data points.

        Args:
            features: Features to predict the class for.

        Returns:
            The predicted class for each data point.
        """
        if self._unique_cls is None:
            return self.model.predict(features)
        else:
            return np.ones(len(features), dtype=int) * self._unique_cls

    def score(self, features: NDArray[np.float_], labels: NDArray[np.int_]) -> float:
        """
        Proxy method for the `score` method of the wrapped model.

        Args:
            features: Features to fit the model on.
            labels: Labels to fit the model on.

        Returns:
            The score of the features and labels according to the wrapped model.
        """
        return self.model.score(features, labels)
