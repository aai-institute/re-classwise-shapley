import math as m
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from pydvl.utils import Dataset, ensure_seed_sequence, maybe_progress
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import Seed

logger = setup_logger()


def principal_resnet_components(
    x: NDArray[np.float_],
    y: NDArray[np.int_],
    n_components: int,
    grayscale: bool = False,
    seed: Seed = None,
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """
    This method calculates an internal feature representation by using a pre-trained
    resnet18. Subsequently, the internal representation is used by PCA to extract the
    main principal components.

    :param x: Input features of the data. The data is expected to be in the shape of
        (n_samples, n_features) or (n_samples, n_channels, n_features). The former case
        represents grayscale images and the latter full color images.
    :param y: Output feature of the data. The output features are expected to be of
        shape (n_samples,) and type int.
    :param n_components: The number of principal components (See PCA).
    :param grayscale: True if the input data is grayscale. In this case, the single
        input channel is replicated to red, green and blue channels without scaling the
        grayscale values.
    :param seed: The seed to use for the random number generator.
    :returns: A tuple containing the processed input and passed output features.
    """
    x = _calculate_resnet18_features(x, grayscale)
    x = _extract_principal_components(x, n_components, seed)
    return x, y


def _calculate_resnet18_features(
    x: NDArray[np.float_],
    grayscale: bool,
    batch_size: int = 100,
    progress: bool = True,
) -> NDArray[np.float_]:
    """
    This method calculates an internal feature representation by using a pre-trained
    resnet18. All images are processed in batched to avoid memory issues.
    :param x: Input features of the data. The data is expected to be in the shape of
        (n_samples, n_features) or (n_samples, n_channels, n_features). The former case
        represents grayscale images and the latter full color images.
    :param grayscale: True if the input data is grayscale. In this case, the single
        input channel is replicated to red, green and blue channels without scaling the
        grayscale values.
    :param batch_size: Number of images which are processed in a single batch.
    :param progress: Whether to display a progress bar.
    :returns: Processed features.
    """
    logger.info("Applying resnet18.")
    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    preprocess = weights.transforms()

    collected_features = []
    num_batches = int(m.ceil(len(x) / batch_size))
    for batch_num in maybe_progress(
        range(num_batches), display=progress, desc="Processing batches"
    ):
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
    return features


def _extract_principal_components(
    x: NDArray[np.float_], n_components: int, seed: Seed = None
) -> NDArray[np.float_]:
    """
    This method extracts the main principal components from the given features. Before
    and after applying PCA the features are scaled to have zero mean and unit variance.
    :param x: Input features of the data. The data is expected to be in the shape of
        (n_samples, n_features).
    :param n_components: Number of principal components to be extracted.
    :param seed: The seed to use for the random number generator.
    :returns: A tuple containing the passed input and processed output features.
    """
    logger.info(f"Fitting PCA with {n_components} components.")
    random_state = ensure_seed_sequence(seed).generate_state(1)[0]
    pca = PCA(n_components=n_components, random_state=random_state)
    x = (x - x.mean()) / x.std()
    x = pca.fit_transform(x)
    return (x - x.mean()) / x.std()


def threshold_y(
    x: NDArray[np.float_], y: NDArray[np.float_], threshold: int, seed: Seed = None
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """
    Leave x as it is. All values of y which are smaller or equal to the threshold
    are set to 0 and all values which are larger are set to 1.

    :param x: Input features of the data. The data is expected to be in the shape of
        (n_samples, n_features).
    :param y: Output feature of the data. The output features are expected to be of
        shape (n_samples,) and type int.
    :param threshold: Threshold for defining binary classes for y.
    :param seed: Unused.
    :returns: A tuple containing the processed input and passed output features.
    """
    y = (y <= threshold).astype(int)
    return x, y


PreprocessorRegistry = {
    "principal_resnet_components": principal_resnet_components,
    "threshold_y": threshold_y,
}


def flip_labels(
    dataset: Dataset, perc: float = 0.2, seed: Seed = None
) -> Tuple[NDArray[int], Dict]:
    labels = dataset.y_train
    rng = np.random.default_rng(seed)
    num_data_indices = int(perc * len(labels))
    p = rng.permutation(len(labels))[:num_data_indices]
    labels[p] = 1 - labels[p]
    dataset = deepcopy(dataset)
    dataset.y_train = labels
    return dataset, {"idx": [int(i) for i in p], "n_flipped": num_data_indices}
