# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Name -> class registries for backends and methods.

`--backend pangu --method jointfix` resolves through here. Register with the
decorators so adding a backend/method is one line + the new file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Type

if TYPE_CHECKING:  # avoid a runtime cycle: registry is a leaf module
    from jointfix.backends.base import ModelBackend
    from jointfix.methods.base import QuantMethod

_BACKENDS: Dict[str, "Type[ModelBackend]"] = {}
_METHODS: Dict[str, "Type[QuantMethod]"] = {}


def register_backend(name: str) -> Callable[[Type[ModelBackend]], Type[ModelBackend]]:
    def deco(cls: Type[ModelBackend]) -> Type[ModelBackend]:
        _BACKENDS[name] = cls
        return cls
    return deco


def register_method(name: str) -> Callable[[Type[QuantMethod]], Type[QuantMethod]]:
    def deco(cls: Type[QuantMethod]) -> Type[QuantMethod]:
        _METHODS[name] = cls
        return cls
    return deco


def get_backend(name: str) -> Type[ModelBackend]:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def get_method(name: str) -> Type[QuantMethod]:
    if name not in _METHODS:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(_METHODS)}")
    return _METHODS[name]


def available_backends() -> list:
    return sorted(_BACKENDS)


def available_methods() -> list:
    return sorted(_METHODS)
