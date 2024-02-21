import numbers
import warnings
from typing import Tuple

import numpy as np
from numpy._typing import NDArray
from pydvl.utils import Utility, random_powerset
from pydvl.value import ValuationStatus
from pydvl.value.results import ValuationResult

__all__ = ["class_wise_shapley"]

from itertools import chain, combinations


def powerset_wo_null(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(1, len(xs) + 1))


def class_wise_shapley(
    u: Utility, *, progress: bool = True, num_sets: int = 500, eps: float = 1e-4
) -> ValuationResult:
    r"""Computes the class-wise Shapley value using the formulation with permutations:

    :param u: Utility object with model, data, and scoring function
    :param progress: Whether to display progress bars for each job.
    :param num_sets: The number of sets to use in the truncated monte carlo estimator.
    :param eps: The threshold when updating using the truncated monte carlo estimator.
    :return: Object with the data values.
    """

    if not all(map(lambda v: isinstance(v, numbers.Integral), u.data.y_train)):
        raise ValueError("The supplied dataset has to be a classification dataset.")

    n_train = len(u.data)
    y_train = u.data.y_train
    values = np.zeros(n_train)

    unique_labels = np.unique(y_train)
    for idx_label, label in enumerate(unique_labels):
        active_elements = y_train == label
        label_set = np.where(active_elements)[0]
        complement_label_set = np.where(~active_elements)[0]
        last_permutation_label_set = np.arange(len(label_set))

        for num_subset, subset_complement in enumerate(
            random_powerset(complement_label_set, max_subsets=num_sets)
        ):
            permutation_label_set = np.random.permutation(label_set)
            train_set = np.concatenate((label_set, subset_complement))
            in_cls_acc, out_of_cls_acc = estimate_data_value(u, train_set, label)

            data_value = 0
            final_data_value = in_cls_acc * np.exp(out_of_cls_acc)

            for j in range(len(label_set)):
                if np.abs(final_data_value - data_value) < eps:
                    next_data_value = data_value

                else:
                    train_set = np.concatenate(
                        (permutation_label_set[: j + 1], subset_complement)
                    )
                    in_cls_acc, out_of_cls_acc = estimate_data_value(
                        u, train_set, label
                    )
                    next_data_value = in_cls_acc * np.exp(out_of_cls_acc)

                diff_data_value = next_data_value - data_value
                values[permutation_label_set[j]] = (
                    num_subset * values[last_permutation_label_set[j]] + diff_data_value
                ) / (num_subset + 1)
                data_value = next_data_value

            last_permutation_label_set = permutation_label_set

        sigma_c = np.sum(values[label_set])
        in_cls_acc, _ = estimate_data_value(u, np.arange(n_train), label)
        values[label_set] *= in_cls_acc / sigma_c

    return ValuationResult(
        algorithm="class_wise_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )


def estimate_data_value(
    u: Utility, indices: NDArray[np.int_], label: np.int_
) -> Tuple[float, float]:
    x_train_s = u.data.x_train[indices]
    y_train_s = u.data.y_train[indices]
    u.model.fit(x_train_s, y_train_s)

    x_dev = u.data.x_test
    y_dev = u.data.y_test
    y_dev_pred = u.model.predict(x_dev)
    y_dev_matched = y_dev_pred == y_dev
    cls_idx_dev = np.where(y_dev == label)[0]
    n_dev = len(x_dev)
    inv_cls_idx_dev = invert_idx(cls_idx_dev, n_dev)

    acc_in_cls = np.sum(y_dev_matched[cls_idx_dev]) / n_dev
    acc_out_of_cls = np.sum(y_dev_matched[inv_cls_idx_dev]) / n_dev
    return acc_in_cls, acc_out_of_cls


def invert_idx(p: np.ndarray, n: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[p] = 0
    return np.arange(n)[mask]
