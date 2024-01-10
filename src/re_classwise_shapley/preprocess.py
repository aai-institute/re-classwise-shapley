import math as m
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydvl.utils import Dataset, ensure_seed_sequence
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from re_classwise_shapley.filter import FilterRegistry
from re_classwise_shapley.log import setup_logger
from re_classwise_shapley.types import RawDataset, Seed

__all__ = ["PreprocessorRegistry", "apply_sample_preprocessors", "preprocess_dataset"]

logger = setup_logger(__name__)


def principal_resnet_components(
    x: NDArray[np.float_],
    y: NDArray[np.int_],
    n_components: int,
    grayscale: bool = False,
    seed: Seed = None,
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """
    This method calculates an internal feature representation by using a pre-trained
    resnet18. All images are processed in batched to avoid memory issues. Afterward,
    the main principal components are extracted from the features.

    Args:
        x: Input features of the data. The data is expected to be in the shape of
            (n_samples, n_features) or (n_samples, n_channels, n_features). The former
            case represents grayscale images and the latter full color images.
        y: Output feature of the data. The output features are expected to be of
            shape (n_samples,) and type int.
        n_components: The number of principal components (See PCA).
        grayscale: True if the input data is grayscale. In this case, the single
            input channel is replicated to red, green and blue channels without scaling
            the grayscale values.
        seed: Either a seed or a seed sequence to use for the random number generator.

    Returns:
        A tuple containing the processed input and passed output features.
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

    Args:
        x: Input features of the data. The data is expected to be in the shape of
            (n_samples, n_features) or (n_samples, n_channels, n_features). The former
             case represents grayscale images and the latter full color images.
        grayscale: True if the input data is grayscale. In this case, the single input
            channel is replicated to red, green and blue channels without scaling the
            grayscale values.
        batch_size: Number of images which are processed in a single batch.
        progress: Whether to display a progress bar.

    Returns:
        Processed features.
    """
    logger.info("Applying resnet18.")
    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    preprocess = weights.transforms()

    collected_features = []
    num_batches = int(m.ceil(len(x) / batch_size))
    for batch_num in tqdm(
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

    Args:
        x: Input features of the data. The data is expected to be in the shape of
            (n_samples, n_features).
        n_components: Number of principal components to be extracted.
        seed: Either a seed or a seed sequence to use for the random number generator.

    Returns:
        A tuple containing the passed input and processed output features.
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

    Args:
        x: Input features of the data. The data is expected to be in the shape of
            (n_samples, n_features).
        y: Output feature of the data. The output features are expected to be of
            shape (n_samples,) and type int.
        threshold: Threshold for defining binary classes for y.
        seed: Unused.
    Returns:
         A tuple containing the processed input and passed output features.
    """
    y = (y <= threshold).astype(int)
    return x, y


PreprocessorRegistry = {
    "principal_resnet_components": principal_resnet_components,
    "threshold_y": threshold_y,
}


def preprocess_dataset(raw_dataset: RawDataset, dataset_kwargs: Dict) -> RawDataset:
    """
    Preprocesses a dataset and returns preprocessed data.

    Args:
        raw_dataset: The raw dataset to preprocess.
        dataset_kwargs: The dataset kwargs for processing. Contains the keys `filters`
            and `preprocessor`. The `filters` key contains a dictionary of filters to
            apply. The `preprocessor` key contains a dictionary of preprocessors to
            apply.

    Returns:
        The preprocessed dataset as a tuple of x, y and additional info. Additional
        information contains a mapping from file_names to dictionaries (to be saved as
        `*.json`). It contains a file name `info.json` with information `feature_names`,
        `target_names` and `description`. It also contains a file name `filters.json`
        with the applied filters and a file name `preprocess.json` with the applied
        preprocessors.
    """
    x, y, additional_info = raw_dataset

    filters = dataset_kwargs.get("filters", None)
    if filters is not None:
        for filter_name, filter_kwargs in filters.items():
            logger.info(f"Applying filter '{filter_name}'.")
            data_filter = FilterRegistry[filter_name]
            x, y = data_filter(x, y, **filter_kwargs)

    logger.info("Applying preprocessors.")
    preprocessor_definitions = dataset_kwargs.pop("preprocessor", None)
    if preprocessor_definitions is not None:
        for (
            preprocessor_name,
            preprocessor_kwargs,
        ) in preprocessor_definitions.items():
            logger.info(f"Applying preprocessor '{preprocessor_name}'.")
            preprocessor = PreprocessorRegistry[preprocessor_name]
            x, y = preprocessor(x, y, **preprocessor_kwargs)

    logger.info("Encoding labels to integers.")
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    additional_info["info.json"]["label_distribution"] = (
        pd.value_counts(y) / len(y)
    ).to_dict()
    additional_info["filters.json"] = filters
    additional_info["preprocess.json"] = preprocessor_definitions
    return x, y, additional_info


def apply_sample_preprocessors(
    dataset: Dataset, preprocessor_configs: Dict, seed: List[Seed]
) -> Tuple[Dataset, Dict]:
    """
    Applies a list of preprocessors (specified by `preprocessor_configs`) to a dataset.
    `preprocessor_configs` is a dictionary containing the name of the preprocessor as
    key and the configuration as value. The configuration is passed to the preprocessor
    generator function obtained from the `PreprocessorRegistry`.

    Args:
        dataset: Dataset to preprocess.
        preprocessor_configs: A dictionary containing the configurations of the
            preprocessors.
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        A tuple containing the preprocessed dataset and a dictionary containing
    """

    preprocess_info = {}
    for idx, (preprocessor_name, preprocessor_config) in enumerate(
        preprocessor_configs.items()
    ):
        preprocessor_fn = SamplePreprocessorRegistry[preprocessor_name]
        dataset, info = preprocessor_fn(dataset, **preprocessor_config, seed=seed[idx])
        preprocess_info.update(
            {f"preprocessor.{preprocessor_name}.{k}": v for k, v in info.items()}
        )

    return dataset, preprocess_info


def flip_labels(
    dataset: Dataset, perc: float = 0.2, seed: Seed = None
) -> Tuple[Dataset, Dict]:
    """
    Flips a percentage of labels in a dataset. The labels are flipped randomly. The
    number of flipped labels is returned in the `preprocess_info` dictionary.

    Args:
        dataset: Dataset to flip labels.
        perc: Number of labels to flip in percent. Must be in the range [0, 1].
        seed: Either an instance of a numpy random number generator or a seed for it.

    Returns:
        A tuple containing the dataset with flipped labels and a dictionary containing
        the number and indices of the flipped labels.

    """
    labels = dataset.y_train
    rng = np.random.default_rng(seed)
    num_data_indices = int(perc * len(labels))
    p = rng.permutation(len(labels))[:num_data_indices]
    labels[p] = 1 - labels[p]
    dataset.y_train = labels
    return dataset, {"idx": [int(i) for i in p], "n_flipped": num_data_indices}


SamplePreprocessorRegistry = {"flip_labels": flip_labels}
