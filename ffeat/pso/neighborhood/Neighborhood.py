###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t
from ffeat import Pipe


class Neighborhood(Pipe):
    """
    Base class to define PSO neighborhood.
    """

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        This method should define the neighborhood and should be implemented by the derived class.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        The tensor should be of type `torch.long`.
        """
        raise NotImplementedError()

    def _handle_size(self, size, pop_size):
        """
        Handles size of the population in case the size is float or integer.
        :param size: Expected size, either as float or as integer.
        :param pop_size: Size of the population.
        :return: Size in absolute number.
        """
        if int(size) != size:
            if size <= 1.0:
                return int(pop_size * size)
        return int(size)
