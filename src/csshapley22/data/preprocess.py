from typing import Tuple, Union

import numpy as np
import torch
from numpy._typing import NDArray
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

    features = (features - features.mean()) / features.std()
    logger.info("Fitting PCA.")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


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


PreprocessorRegistry = {"principal_resnet_components": principal_resnet_components}

FilterRegistry = {"binarization": binarize_classes}
