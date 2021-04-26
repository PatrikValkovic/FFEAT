###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Union, Callable, Tuple, Any, Dict
import math
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class RankScale(Pipe):
    def __init__(self,
                 minimum: Union[_IFU, Callable[..., _IFU]],
                 maximum: Union[_IFU, Callable[..., _IFU]]):
        self._minimum = self._handle_parameter(minimum)
        self._maximum = self._handle_parameter(maximum)

    def __call__(self, fitnesses, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        min = self._minimum(fitnesses, *args, **kwargs)
        if not isinstance(min, float):
            min = float(min)
        max = self._maximum(fitnesses, *args, **kwargs)
        if not isinstance(max, float):
            max = float(max)

        num = len(fitnesses)
        step = (max - min) / num
        order = t.argsort(fitnesses)
        tmpfitnes = t.arange(0, num, device=fitnesses.device, dtype=fitnesses.dtype)
        tmpfitnes.multiply_(step).add_(min)
        newfitness = t.empty_like(fitnesses)
        newfitness[order] = tmpfitnes
        kwargs['new_fitness'] = newfitness

        return (newfitness, *args), kwargs