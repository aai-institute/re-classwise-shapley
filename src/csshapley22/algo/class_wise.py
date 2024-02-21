import warnings
from typing import Tuple

import numpy as np
from pydvl.utils import Utility
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["class_wise_shapley"]

from itertools import chain, combinations


def powerset_wo_null(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(1, len(xs) + 1))


def class_wise_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    r"""Computes the class-wise Shapley value using the formulation with permutations:

    :param u: Utility object with model, data, and scoring function
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """

    if u.data.y_train.dtype != np.int64:
        raise ValueError("The supplied dataset has to be a classification dataset.")

    x_train = u.data.x_train
    y_train = u.data.y_train
    n_train = len(x_train)

    # Note that the cache in utility saves most of the refitting because we
    # use frozenset for the input.
    if n_train > 10:
        warnings.warn(
            f"Large dataset! Computation requires {n_train}! calls to utility()",
            RuntimeWarning,
        )

    values = np.zeros(n_train)
    num_sets = 500
    eps = 1e-4

    all_cls = np.unique(y_train)
    for cls_num, cls_val in enumerate(all_cls):
        cls_idx_train = np.where(y_train == cls_val)[0]
        inv_cls_idx_train = invert_idx(cls_idx_train, n_train)

        for k in range(1, num_sets + 1):
            cls_idx_perm_train = np.random.permutation(cls_idx_train)
            subset_length = np.random.randint(1, len(inv_cls_idx_train) + 1)
            inv_cls_idx_subset_train = np.random.permutation(inv_cls_idx_train)[
                :subset_length
            ]

            idx_train_all = np.concatenate((cls_idx_train, inv_cls_idx_subset_train))
            v_its = np.empty(len(cls_idx_train) + 1)
            in_cls_acc, out_of_cls_acc = estimate_data_value(u, idx_train_all, cls_val)
            v_its[0] = 0
            v_its[-1] = in_cls_acc * np.exp(out_of_cls_acc)

            for j in range(len(cls_idx_train)):
                if np.abs(v_its[-1] - v_its[j]) < eps:
                    v_its[j + 1] = v_its[j]

                else:
                    idx_train_all = np.concatenate(
                        (cls_idx_perm_train[: j + 1], inv_cls_idx_subset_train)
                    )
                    in_cls_acc, out_of_cls_acc = estimate_data_value(
                        u, idx_train_all, cls_val
                    )
                    v_its[j + 1] = in_cls_acc * np.exp(out_of_cls_acc)

                values[cls_idx_perm_train[j]] = (k - 1) / k * values[
                    cls_idx_perm_train[j]
                ] + 1 / k * (v_its[j + 1] - v_its[j])

        sigma_c = np.sum(values[cls_idx_train])
        in_cls_acc, _ = estimate_data_value(u, np.arange(n_train), cls_val)
        values[cls_idx_train] *= in_cls_acc / sigma_c  # TODO Remove divisor hack

    return ValuationResult(
        algorithm="class_wise_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )


def estimate_data_value(
    u: Utility, idx_train_all: np.ndarray, cls_val: int
) -> Tuple[float, float]:
    x_train_s = u.data.x_train[idx_train_all]
    y_train_s = u.data.y_train[idx_train_all]
    u.model.fit(x_train_s, y_train_s)

    x_dev = u.data.x_test
    y_dev = u.data.y_test
    y_dev_pred = u.model.predict(x_dev)
    y_dev_matched = y_dev_pred == y_dev
    cls_idx_dev = np.where(y_dev == cls_val)[0]
    n_dev = len(x_dev)
    inv_cls_idx_dev = invert_idx(cls_idx_dev, n_dev)

    acc_in_cls = np.sum(y_dev_matched[cls_idx_dev]) / n_dev
    acc_out_of_cls = np.sum(y_dev_matched[inv_cls_idx_dev]) / n_dev
    return acc_in_cls, acc_out_of_cls


def invert_idx(p: np.ndarray, n: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[p] = 0
    return np.arange(n)[mask]
