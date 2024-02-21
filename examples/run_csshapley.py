import logging

import numpy as np
from pydvl.utils import Utility
from pydvl.utils.dataset import Dataset, synthetic_classification_dataset
from sklearn.linear_model import LogisticRegression

from csshapley22.algo.class_wise import class_wise_shapley

log = logging.getLogger(__name__)


def run_classwise_shapley():
    """Compares the combinatorial exact shapley and permutation exact shapley with
    the analytic_shapley calculation for a dummy model.
    """
    num_samples = 100
    sigma = 0.2
    means = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    train_data, val_data, test_data = synthetic_classification_dataset(
        means, sigma, num_samples, train_size=0.7, test_size=0.2
    )

    model = LogisticRegression()
    dataset = Dataset(*train_data, *val_data)
    u = Utility(model=model, data=dataset)
    valuation_result = class_wise_shapley(u)
    print(valuation_result.values)


if __name__ == "__main__":
    run_classwise_shapley()
