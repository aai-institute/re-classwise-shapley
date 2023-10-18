from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


def binarize_classes(
    features: NDArray[np.float_],
    labels: Union[NDArray[np.float_], NDArray[np.int_]],
    label_zero: Union[str, int],
    label_one: Union[str, int],
    shuffle: bool = True,
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """
    Binarize the labels of a multi-class dataset by filtering out all other labels.

    Args:
        features: Features of the dataset.
        labels: Labels of the dataset.
        label_zero: Label which is mapped to 0.
        label_one: Label which is mapped to 1.
        shuffle: Whether to shuffle the data after binarization.

    Returns:
        A tuple containing the selected input and output features.
    """
    idx_label_zero = np.where(labels == label_zero)[0]
    idx_label_one = np.where(labels == label_one)[0]
    labels[idx_label_zero] = 0
    labels[idx_label_one] = 1
    idx_labels = np.concatenate((idx_label_zero, idx_label_one))

    if shuffle:
        idx_labels = idx_labels[np.random.permutation(len(idx_labels))]

    features = features[idx_labels]
    labels = labels[idx_labels]
    return features, labels


FilterRegistry = {"binarization": binarize_classes}
