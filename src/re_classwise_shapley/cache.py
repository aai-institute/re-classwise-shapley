from typing import Any, Optional

from pydvl.utils import MemcachedCacheBackend


class PrefixedMemcachedCacheBackend(MemcachedCacheBackend):
    def __init__(self, *args, prefix: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = prefix

    def get(self, key: str) -> Optional[Any]:
        return super().get(f"{self._prefix}/{key}")

    def set(self, key: str, value: Any) -> None:
        super().set(f"{self._prefix}/{key}", value)
