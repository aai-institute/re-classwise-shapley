from typing import Callable, Dict, Tuple

from pydvl.utils import Dataset, SupervisedModel, Utility
from pydvl.value import ValuationResult

"""
K: dataset_name
V: A tuple of a validation and test set. Both shall contain the same training data.
"""
ValTestSetFactory = Dict[str, Tuple[Dataset, Dataset]]

"""
K: valuation_method_name
V: A estimator mapping the utility to an valuation_result. Both shall contain the same training data.
"""
BaseValueEstimator = Callable[[Utility], ValuationResult]
ValuationMethodsFactory = Dict[str, BaseValueEstimator]

"""
K: model_name
V: A callable which throws a super vised model out of it.
"""
ModelGenerator = Callable[[], SupervisedModel]
ModelGeneratorFactory = Dict[str, ModelGenerator]
