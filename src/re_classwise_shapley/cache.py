from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, cast

from pydvl.utils import MemcachedCacheBackend, MemcachedClientConfig


class PrefixedMemcachedCacheBackend(MemcachedCacheBackend):
    def __init__(
        self, config: MemcachedClientConfig = MemcachedClientConfig(), *, prefix: str
    ):
        self._prefix = prefix
        super().__init__(config)

    def get(self, key: str) -> Optional[Any]:
        return super().get(f"{self._prefix}/{key}")

    def set(self, key: str, value: Any) -> None:
        super().set(f"{self._prefix}/{key}", value)

    def __getstate__(self) -> Dict:
        """Enables pickling after a socket has been opened to the
        memcached server, by removing the client from the stored
        data."""
        odict = super().__getstate__()
        odict["_prefix"] = self._prefix
        return odict

    def __setstate__(self, d: Dict):
        """Restores a client connection after loading from a pickle."""
        self._prefix = d.pop("_prefix")
        super().__setstate__(d)
