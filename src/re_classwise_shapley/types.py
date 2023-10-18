from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ["RawDataset", "Seed", "OneOrMany", "ensure_list"]


RawDataset = Tuple[
    NDArray[np.float_], Union[NDArray[np.float_], NDArray[np.int_]], Dict[str, Dict]
]
Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]


T = TypeVar("T")
OneOrMany = Union[T, List[T]]


def ensure_list(x: OneOrMany[T]) -> List[T]:
    """
    Converts an input to a list. If the input is already a list, it is returned
    unchanged. Otherwise, the element is wrapped in a list.

    Args:
        x: Input to be converted to a list.

    Returns:
        Either the input list or a list containing the input.
    """
    return x if isinstance(x, List) else [x]
