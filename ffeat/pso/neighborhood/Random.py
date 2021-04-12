###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Union, Callable
import torch as t
from .Neighborhood import Neighborhood


class Random(Neighborhood):
    def __init__(self,
                 size: Union[float, int, Callable[..., Union[int, float]]]):
        self._size = self._handle_parameter(size)

    def __call__(self, fitnesses, positions, **kwargs) -> t.Tensor:
        pop_size = len(fitnesses)
        neighbors = self._handle_size(self._size(fitnesses, positions, **kwargs), pop_size)

        neigborhood = t.randint(pop_size, (pop_size, neighbors), dtype=t.long, device=fitnesses.device)
        return neigborhood
