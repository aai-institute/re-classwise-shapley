from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

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
    Ensures that the input is a list.
    Args:
        x: Input to convert to a list.

    Returns:
        List of the input.
    """
    return x if not isinstance(input, Sequence) else [x]
