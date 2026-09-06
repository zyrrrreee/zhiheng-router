"""Order-independent namespace random streams derived with SHA256."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import TypeVar


T = TypeVar("T")


class NamespaceRNG:
    """Derive a fresh deterministic RNG for each namespace and stable object ID."""

    def __init__(self, master_seed: int, namespaces: Mapping[str, str]) -> None:
        if type(master_seed) is not int or master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if not namespaces or any(not isinstance(k, str) or not k.strip()
                                 or not isinstance(v, str) or not v.strip()
                                 for k, v in namespaces.items()):
            raise ValueError("namespaces must map nonblank names to nonblank salts")
        if len(set(namespaces.values())) != len(namespaces):
            raise ValueError("namespace salts must be unique")
        self.master_seed = master_seed
        self.namespaces = dict(namespaces)

    def derive_seed(self, namespace: str, stable_id: str, *, stream: str = "default") -> int:
        if namespace not in self.namespaces:
            raise ValueError(f"unknown RNG namespace: {namespace}")
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError("stable_id must be nonblank")
        if not isinstance(stream, str) or not stream.strip():
            raise ValueError("stream must be nonblank")
        material = "\0".join((str(self.master_seed), self.namespaces[namespace], stable_id, stream))
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:16], "big")

    def random(self, namespace: str, stable_id: str, *, stream: str = "default") -> random.Random:
        return random.Random(self.derive_seed(namespace, stable_id, stream=stream))

    def choice(self, namespace: str, stable_id: str, values: Sequence[T], *,
               stream: str = "default") -> T:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return self.random(namespace, stable_id, stream=stream).choice(values)

    def sample(self, namespace: str, stable_id: str, values: Sequence[T], count: int, *,
               stream: str = "default") -> list[T]:
        if type(count) is not int or count < 0 or count > len(values):
            raise ValueError("invalid sample count")
        return self.random(namespace, stable_id, stream=stream).sample(list(values), count)
