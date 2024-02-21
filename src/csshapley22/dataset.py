import numpy as np
from numpy.typing import NDArray
from pydvl.utils.dataset import Dataset
from sklearn.model_selection import train_test_split

__all__ = ["create_synthetic_dataset"]


def flip_labels(
    y: NDArray[np.int_], percentage: float, *, random_state: np.random.RandomState
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = random_state.choice(
        np.arange(len(y)), size=int(percentage * len(y)), replace=False
    )
    y = y.copy()
    y[indices] = np.logical_not(y[indices])
    return y, indices


def create_synthetic_dataset(
    n_features: int,
    n_train_samples: int,
    n_test_samples: int,
    n_val_samples: int = 0,
    *,
    random_state: np.random.RandomState | int | None = None,
) -> Dataset | tuple[Dataset, Dataset]:
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_total_samples = n_train_samples + n_val_samples + n_test_samples
    X = random_state.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_total_samples,
    )
    feature_mask = random_state.random(n_features) > 0.5
    Xb = X @ feature_mask
    Xb -= np.mean(X)
    pr = 1 / (1 + np.exp(-Xb))
    y = random_state.binomial(n=1, p=pr).astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        X,
        y,
        train_size=n_train_samples + n_val_samples,
        stratify=y,
        random_state=random_state,
    )

    if n_val_samples == 0:
        x_train = x_train_val
        y_train = y_train_val
        dataset = Dataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        return dataset

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        train_size=n_train_samples,
        stratify=y_train_val,
        random_state=random_state,
    )

    train_dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_val,
        y_test=y_val,
    )

    test_dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    return train_dataset, test_dataset
