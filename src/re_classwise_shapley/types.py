from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

ValuationMethodDict = Dict[str, Callable]
FloatIntStringArray = Union[NDArray[np.float_], NDArray[np.int_]]
RawDataset = Tuple[NDArray[np.float_], FloatIntStringArray, Dict[str, Dict]]
Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
