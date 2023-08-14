from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from re_classwise_shapley.types import FloatIntStringArray


def binarize_classes(
    x: NDArray[np.float_],
    y: FloatIntStringArray,
    label_zero: Union[str, int],
    label_one: Union[str, int],
    shuffle: bool = True,
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """
    Binarize the labels of a multi-class dataset by filtering out all other labels.
    :param x: Input features of the data. The data is expected to be in the shape of
        (n_samples, n_features).
    :param y: Output feature of the data. The output features are expected to be of
        shape (n_samples,) and type int.
    :param label_zero: Label which is mapped to 0.
    :param label_one: Label which is mapped to 1.
    :param shuffle: Whether to shuffle the data after binarization.
    :return: A tuple containing the selected input and output features.
    """
    idx_zero = np.where(y == label_zero)[0]
    idx_one = np.where(y == label_one)[0]
    y[idx_zero] = 0
    y[idx_one] = 1
    all_idx = np.concatenate((idx_zero, idx_one))

    x = x[all_idx]
    y = y[all_idx]
    if shuffle:
        p = np.random.permutation(len(x))
        x, y = x[p], y[p]

    return x, y


FilterRegistry = {"binarization": binarize_classes}
