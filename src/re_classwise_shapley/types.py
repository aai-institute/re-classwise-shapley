from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydvl.utils import Dataset, SupervisedModel, Utility
from pydvl.value import ValuationResult

"""
K: dataset_name
V: A tuple of a validation and test set. Both shall contain the same training data.
"""
ValTestSetFactory = Dict[str, Callable[[], Tuple[Dataset, Dataset]]]

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

FloatIntStringArray = Union[NDArray[np.float_], NDArray[np.int_]]
RawDataset = Tuple[NDArray[np.float_], FloatIntStringArray, Dict[str, Dict]]
Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
