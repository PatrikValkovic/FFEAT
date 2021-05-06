###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION

_IFU = Union[int, float]


class _Velocity(Pipe):
    """
    Base class for particles velocity clipping.
    """
    pass


class VelocityValue(_Velocity):
    """
    Clip each dimension of velocity by value.
    """
    def __init__(self, min: _IFU, max: _IFU):
        """
        Clip each dimension of velocity by value.
        :param min: Minimum velocity in each dimension.
        :param max: Maximum velocity in each dimension.
        """
        self._min = min
        self._max = max

    def __call__(self, velocities, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Clips velocities and return them.
        :param velocities: Current velocities.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Update velocities with given arguments.
        """
        velocities.clip_(self._min, self._max)
        return (velocities, *args), kwargs


class VelocityNorm(_Velocity):
    """
    Clip size of the velocity. Particles can't move faster.
    """
    def __init__(self, max: _IFU):
        """
        Clip size of the velocity. Particles can't move faster.
        :param max: Maximum velocity size.
        """
        self._max = max

    def __call__(self, velocities, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Clips velocities and return them.
        :param velocities: Current velocities.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Update velocities with given arguments.
        """
        dims = len(velocities.shape[1:])
        norms = t.norm(velocities, dim=list(range(1,dims+1)))
        not_exceeds = norms < self._max
        norms.divide_(self._max)
        norms[not_exceeds] = 1
        velocities.divide_(
            norms.reshape(len(velocities), *([1] * dims))
        )
        return (velocities, *args), kwargs
