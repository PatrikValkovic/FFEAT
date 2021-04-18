###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class _Velocity(Pipe):
    pass


class VelocityValue(_Velocity):
    def __init__(self, min: _IFU, max: _IFU):
        self._min = min
        self._max = max

    def __call__(self, velocities, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        velocities.clip_(self._min, self._max)
        return (velocities, *args), kwargs


class VelocityNorm(_Velocity):
    def __init__(self, max: _IFU):
        self._max = max

    def __call__(self, velocities, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        dims = len(velocities.shape[1:])
        norms = t.norm(velocities, dim=list(range(1,dims+1)))
        not_exceeds = norms < self._max
        norms.divide_(self._max)
        norms[not_exceeds] = 1
        velocities.divide_(
            norms.reshape(len(velocities), *([1] * dims))
        )
        return (velocities, *args), kwargs
