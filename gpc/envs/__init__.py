"""Training environments and shared simulator types.

Concrete env classes are loaded lazily so ``from gpc.envs import TrainingEnv``
(or importing ``gpc.training``) does not import every Hydrax demo task.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import SimulatorState, TrainingEnv

__all__ = [
    "SimulatorState",
    "TrainingEnv",
    "CartPoleEnv",
    "CraneEnv",
    "DoubleCartPoleEnv",
    "ParticleEnv",
    "PendulumEnv",
    "PushTEnv",
    "WalkerEnv",
    "HumanoidEnv",
    "Locomanip2DEnv",
    "build_fixed_locomanip_task",
]

_LAZY: dict[str, tuple[str, str]] = {
    "CartPoleEnv": (".cart_pole", "CartPoleEnv"),
    "CraneEnv": (".crane", "CraneEnv"),
    "DoubleCartPoleEnv": (".double_cart_pole", "DoubleCartPoleEnv"),
    "HumanoidEnv": (".humanoid", "HumanoidEnv"),
    "ParticleEnv": (".particle", "ParticleEnv"),
    "PendulumEnv": (".pendulum", "PendulumEnv"),
    "PushTEnv": (".pusht", "PushTEnv"),
    "WalkerEnv": (".walker", "WalkerEnv"),
    "Locomanip2DEnv": (".locomanip_2d", "Locomanip2DEnv"),
    "build_fixed_locomanip_task": (".locomanip_2d", "build_fixed_locomanip_task"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
