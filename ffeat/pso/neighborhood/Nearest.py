###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Union, Callable
import torch as t
from .Neighborhood import Neighborhood

_IFU = Union[float, int]

class Nearest(Neighborhood):
    def __init__(self,
                 size: Union[_IFU, Callable[..., _IFU]],
                 norm: int = 2):
        self._size = self._handle_parameter(size)
        self._norm = norm

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        pop_size = len(fitnesses)
        size = self._size(fitnesses, position, **kwargs)
        if int(size) != size:
            if size <= 1.0:
                size = int(pop_size * size)
            else:
                size = int(size)

        distances = t.subtract(position[:,None,:], position[None,:,:])
        distances = t.abs(distances, out=distances)
        distances = t.pow(distances, self._norm, out=distances)
        distances = t.sum(distances, dim=list(range(2, len(position.shape)+1)))
        min = t.argsort(distances, dim=-1)[:,:size]
        return min
