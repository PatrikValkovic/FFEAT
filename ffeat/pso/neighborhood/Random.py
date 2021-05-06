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
    """
    Random neighborhood samples randomly new neighborhood each iteration.
    """
    def __init__(self,
                 size: Union[float, int, Callable[..., Union[int, float]]]):
        """
        Random neighborhood samples randomly new neighborhood each iteration.
        :param size: Size of the neighborhood. May be float (then it is fraction of the original population to select),
        or integer (then it is number of individuals to select).
        """
        self._size = self._handle_parameter(size)

    def __call__(self, fitnesses, positions, **kwargs) -> t.Tensor:
        """
        Creates random neighborhood and returns it.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        """
        pop_size = len(fitnesses)
        neighbors = self._handle_size(self._size(fitnesses, positions, **kwargs), pop_size)

        neigborhood = t.randint(pop_size, (pop_size, neighbors), dtype=t.long, device=fitnesses.device)
        return neigborhood
