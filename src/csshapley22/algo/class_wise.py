"""
Refs:

[1] CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification (https://arxiv.org/abs/2211.06800)
"""

# TODO rename function, and object refs, when transferred to pyDVL.

import numbers
from typing import Optional, Tuple

import numpy as np
from pydvl.utils import Scorer, SupervisedModel, Utility, random_powerset
from pydvl.value import ValuationStatus
from pydvl.value.results import ValuationResult

__all__ = ["class_wise_shapley", "CSScorer"]


def _estimate_in_out_cls_accuracy(
    model: SupervisedModel, x: np.ndarray, labels: np.ndarray, label: np.int_
) -> Tuple[float, float]:
    """
    Estimate the in and out of class accuracy as defined in [1], Equation 3.

    :param model: A model to be used for predicting the labels.
    :param x: The inputs to be used for measuring the accuracies. Has to match the labels.
    :param labels: The labels ot be used for measuring the accuracies. It is divided further by the passed label.
    :param label: The label of the class, which is currently viewed.
    :return: A tuple, containing the in class accuracy as well as the out of class accuracy.
    """
    n = len(x)
    y_pred = model.predict(x)
    label_set_match = labels == label
    label_set = np.where(label_set_match)[0]
    complement_label_set = np.where(~label_set_match)[0]

    y_correct = y_pred == labels
    acc_in_cls = np.sum(y_correct[label_set]) / n
    acc_out_of_cls = np.sum(y_correct[complement_label_set]) / n
    return acc_in_cls, acc_out_of_cls


class CSScorer(Scorer):
    """
    A Scorer object to be used along with the 'class_wise_shapley' function.
    """

    def __init__(self):
        self.label: Optional[np.int_] = None

    def __call__(
        self, model: SupervisedModel, x_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        Estimates the in and out of class accuracies and aggregated them into one float number.
        :param model: A model to be used for predicting the labels.
        :param x_test: The inputs to be used for measuring the accuracies. Has to match the labels.
        :param y_test:  The labels ot be used for measuring the accuracies. It is divided further by the passed label.
        :return: The aggregated number specified by 'in_cls_acc * exp(out_cls_acc)'
        """
        if self.label is None:
            raise ValueError(
                "Please set the label in the class first. By using o.label = <value>."
            )

        in_cls_acc, out_cls_acc = _estimate_in_out_cls_accuracy(
            model, x_test, y_test, self.label
        )
        return in_cls_acc * np.exp(out_cls_acc)


def class_wise_shapley(
    u: Utility, *, progress: bool = True, num_sets: int = 10, eps: float = 1e-4
) -> ValuationResult:
    r"""Computes the class-wise Shapley value using the formulation with permutations:

    :param u: Utility object with model, data, and scoring function. The scoring function has to be of type CSScorer.
    :param progress: Whether to display progress bars for each job.
    :param num_sets: The number of sets to use in the truncated monte carlo estimator.
    :param eps: The threshold when updating using the truncated monte carlo estimator.
    :return: ValuationResult object with the data values.
    """

    if not all(map(lambda v: isinstance(v, numbers.Integral), u.data.y_train)):
        raise ValueError("The supplied dataset has to be a classification dataset.")

    if not isinstance(u.scorer, CSScorer):
        raise ValueError(
            "Please set CSScorer object as scorer object of utility. See scoring argument of Utility."
        )

    n_train = len(u.data)
    y_train = u.data.y_train
    values = np.zeros(n_train)

    unique_labels = np.unique(y_train)
    for idx_label, label in enumerate(unique_labels):
        u.scorer.label = label
        active_elements = y_train == label
        label_set = np.where(active_elements)[0]
        complement_label_set = np.where(~active_elements)[0]
        last_permutation_label_set = np.arange(len(label_set))

        for num_subset, subset_complement in enumerate(
            random_powerset(complement_label_set, max_subsets=num_sets)
        ):
            permutation_label_set = np.random.permutation(label_set)
            train_set = np.concatenate((label_set, subset_complement))
            data_value = 0
            final_data_value = u(train_set)

            for j in range(len(label_set)):
                if np.abs(final_data_value - data_value) < eps:
                    next_data_value = data_value

                else:
                    train_set = np.concatenate(
                        (permutation_label_set[: j + 1], subset_complement)
                    )
                    next_data_value = u(train_set)

                diff_data_value = next_data_value - data_value
                values[permutation_label_set[j]] = (
                    num_subset * values[last_permutation_label_set[j]] + diff_data_value
                ) / (num_subset + 1)
                data_value = next_data_value

            last_permutation_label_set = permutation_label_set

        sigma_c = np.sum(values[label_set])

        # TODO: Phantom call to invoke train
        u(list(range(n_train)))
        in_cls_acc, _ = _estimate_in_out_cls_accuracy(
            u.model, u.data.x_test, u.data.y_test, label
        )

        values[label_set] *= in_cls_acc / sigma_c

    return ValuationResult(
        algorithm="class_wise_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )
