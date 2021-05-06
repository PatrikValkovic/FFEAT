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
    """
    Closest neighbors neighborhood. This neighborhood is very costly.
    """
    def __init__(self,
                 size: Union[_IFU, Callable[..., _IFU]],
                 norm: int = 2):
        """
        Closest neighbors neighborhood. This neighborhood is very costly.
        :param size: Size of the neighborhood. May be float (then it is fraction of the original population to select),
        or integer (then it is number of individuals to select).
        :param norm: Which p-norm to use. By default euclidean norm norm=2.
        """
        self._size = self._handle_parameter(size)
        self._norm = norm

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        Creates closest neighbors neighborhood and returns it. This operation is very costly.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        """
        pop_size = len(fitnesses)
        size = self._handle_size(self._size(fitnesses, position, **kwargs), pop_size)

        distances = t.subtract(position[:,None,:], position[None,:,:])
        distances = t.abs(distances, out=distances)
        distances = t.pow(distances, self._norm, out=distances)
        distances = t.sum(distances, dim=list(range(2, len(position.shape)+1)))
        min = t.argsort(distances, dim=-1)[:,:size]
        return min
