from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

ValuationMethodDict = Dict[str, Callable]
FloatIntStringArray = Union[NDArray[np.float_], NDArray[np.int_]]
RawDataset = Tuple[NDArray[np.float_], FloatIntStringArray, Dict[str, Dict]]
Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]


T = TypeVar("T")
OneOrMany = Union[T, Sequence[T]]


def ensure_list(x: OneOrMany[T]) -> List[T]:
    """
    Converts an input to a list. If the input is already a list, it is returned
    unchanged. Otherwise, the element is wrapped in a list.

    Args:
        x: Input to be converted to a list.

    Returns:
        Either the input list or a list containing the input.
    """
    return x if not isinstance(input, Sequence) else [x]
