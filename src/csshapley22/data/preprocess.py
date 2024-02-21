from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydvl.utils import Dataset
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

from csshapley22.log import setup_logger

logger = setup_logger()


def principal_resnet_components(
    x: np.ndarray, n_components: int, size: int
) -> np.ndarray:
    """
    This method calculates the internal feature representation by a
    pre-trained resnet18. These are then used by PCA to extract the first
    principal components.

    :param x: The features ot be used for processing.
    :param n_components: The number of pca components.
    :return: The transformed values.
    """
    logger.info("Applying resnet18.")
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    x = torch.tensor(x)
    x = torch.cat(3 * [x.reshape([-1, 1, size, size])], axis=1)
    features = resnet(x.type(torch.float))
    features = features.detach().numpy()

    logger.info("Fitting PCA.")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


def dataset_principal_resnet_components(
    val_set: Dataset, test_set: Dataset, n_components: int, size: int
) -> Tuple[Dataset, Dataset]:
    x_features = np.concatenate(
        (val_set.x_train, val_set.x_test, test_set.x_test), axis=0
    )
    pca_x_features = principal_resnet_components(
        x_features, n_components=n_components, size=size
    )
    len_train_set = len(val_set.x_train)
    len_val_set = len(val_set.x_train)
    test_set.x_train = pca_x_features[:len_train_set]
    val_set.x_train = pca_x_features[:len_train_set]
    val_set.x_test = pca_x_features[len_train_set : len_train_set + len_val_set]
    test_set.x_test = pca_x_features[len_train_set + len_val_set]
    return val_set, test_set


def binarize_classes(
    X: np.ndarray,
    y: np.ndarray,
    label_zero: Union[str, int],
    label_one: Union[str, int],
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    idx_zero = np.where(y == label_zero)[0]
    idx_one = np.where(y == label_one)[0]
    y[idx_zero] = 0
    y[idx_one] = 1
    all_idx = np.concatenate((idx_zero, idx_one))

    X = X[all_idx]
    y = y[all_idx]
    if shuffle:
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

    return X, y


PreprocessorRegistry = {
    "principal_resnet_components": dataset_principal_resnet_components
}

FilterRegistry = {"binarization": binarize_classes}
