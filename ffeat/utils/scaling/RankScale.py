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
        order = t.argsort(fitnesses).type(fitnesses.dtype)
        new_fitnesses = t.multiply(order, t.tensor(step), out=order)
        new_fitnesses = t.add(new_fitnesses, min, out=new_fitnesses)
        kwargs['fitness'] = new_fitnesses

        return (new_fitnesses, *args), kwargs