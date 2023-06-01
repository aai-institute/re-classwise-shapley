import math as m
from typing import Tuple, Union

import numpy as np
import torch
from numpy._typing import NDArray
from pydvl.utils import Dataset
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from csshapley22.log import setup_logger

logger = setup_logger()


def principal_resnet_components(
    x: np.ndarray, n_components: int, grayscale: bool = False
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
    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    preprocess = weights.transforms()

    # Apply it to the input image
    collected_features = []
    batch_size = 1000
    num_batches = int(m.ceil(len(x) / batch_size))
    for batch_num in tqdm(range(num_batches), desc="Processing batches"):
        win_x = x[batch_num * batch_size : (batch_num + 1) * batch_size]
        win_x = torch.tensor(win_x).type(torch.float)
        if grayscale:
            win_x = win_x.unsqueeze(1).repeat([1, 3, 1])
        else:
            win_x = win_x.reshape([len(win_x), 3, -1])

        size = int(np.sqrt(win_x.shape[2]))
        win_x = win_x.reshape([len(win_x), 3, size, size])
        pre_win_x = preprocess(win_x)
        features = resnet(pre_win_x)
        features = features.detach().numpy()
        collected_features.append(features)

    features = np.concatenate(tuple(collected_features), axis=0)
    features = (features - features.mean()) / features.std()
    logger.info("Fitting PCA.")
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(features)
    features = (features - features.mean()) / features.std()
    return features


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
