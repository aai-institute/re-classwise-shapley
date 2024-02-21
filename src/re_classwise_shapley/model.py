from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import SupervisedModel
from pydvl.utils.functional import maybe_add_argument
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = ["instantiate_model"]


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
    model_idx = model_kwargs.pop("model")
    model_dict = {
        "gradient_boosting_classifier": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "svm": SVC,
        "mlp_classifier": MLPClassifier,
    }
    model_class = model_dict[model_idx]
    model = maybe_add_argument(model_class, random_state)(
        **model_kwargs, random_state=random_state
    )
    pipeline = make_pipeline(StandardScaler(), model)
    return SingleClassProxySupervisedModel(pipeline)


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
