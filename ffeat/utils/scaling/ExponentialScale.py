###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Union, Callable, Tuple, Any, Dict
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class ExponentialScale(Pipe):
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

        fmax, fmin = t.max(fitnesses), t.min(fitnesses)
        base = pow(max - min + 1, 1 / (fmax - fmin))
        fitnesses = t.sub(fitnesses, fmin, out=fitnesses)
        fitnesses = t.pow(base, fitnesses, out=fitnesses)
        fitnesses = t.add(fitnesses, min - 1, out=fitnesses)
        kwargs['new_fitness'] = fitnesses

        return (fitnesses, *args), kwargs
