###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t

from .Neighborhood import Neighborhood


class Static(Neighborhood):
    """
    Class caching neighborhood returned in the first iteration and returning it afterward.
    """
    def __init__(self, neighborhood: Neighborhood):
        """
        Class caching neighborhood returned in the first iteration and returning it afterward.
        :param neighborhood: Actual neighborhood definition.
        """
        self._neighborhood = neighborhood
        self._indices = None

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        Call the underlying neighborhood and return it. Use this value in the following iterations.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        The tensor should be of type `torch.long`.
        """
        if self._indices is None:
            self._indices = self._neighborhood(fitnesses, position, **kwargs)
        return self._indices
