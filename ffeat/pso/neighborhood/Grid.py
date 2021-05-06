###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Literal, Union, Callable
import torch as t
import numpy as np
from .Neighborhood import Neighborhood
from .Static import Static


class _Grid(Neighborhood):
    """
    Grid neighborhood in arbitrary dimension.
    """
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: Union[float, int, Callable[..., Union[int, float]]],
                 shape: Tuple[int, ...]):
        """
        Grid neighborhood in arbitrary dimension.
        :param type: Type of the neighborhood. So far only linear is supported.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        :param shape: Shape of the topology.
        """
        self._type = type
        self._size = self._handle_parameter(size)
        self._shape = shape

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        Creates grid neighborhood and returns it.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        """
        pop_size = len(fitnesses)
        dimensions = len(self._shape)
        size = self._handle_size(self._size(fitnesses, position, **kwargs), pop_size)
        if pop_size != np.prod(self._shape):
            raise ValueError(f"Grid size doesn't match size of the population, expected {np.prod(self._shape)} individuals")

        orig = np.arange(pop_size)
        if self._type == 'linear':
            indices = np.zeros((pop_size, 2*size*dimensions), dtype=int)
            for dim_i, dim in enumerate(self._shape):
                div_back = np.prod(self._shape[:dim_i])
                div_forw = np.prod(self._shape[:dim_i+1])
                for step in range(size):
                    pos_in_dim = (orig // div_back) % div_forw
                    add_forward = orig // div_forw * div_forw
                    add_backward = orig % div_back
                    neigh_pos_in_dim = (pos_in_dim + step + 1) % dim
                    neigh = neigh_pos_in_dim * div_back + add_forward + add_backward
                    indices[:,step+size*dim_i*2] = neigh
                for step in range(size):
                    pos_in_dim = (orig // div_back) % div_forw
                    add_forward = orig // div_forw * div_forw
                    add_backward = orig % div_back
                    neigh_pos_in_dim = (pos_in_dim + dim - 1 - step) % dim
                    neigh = neigh_pos_in_dim * div_back + add_forward + add_backward
                    indices[:,step+size+size*dim_i*2] = neigh
            return t.tensor(indices, dtype=t.long, device=fitnesses.device)
        # TODO compact and diamond shape
        raise NotImplementedError()


class Grid(Static):
    """
    Grid neighborhood in arbitrary dimension.
    """
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: Union[float, int, Callable[..., Union[int, float]]],
                 shape: Tuple[int, ...]
                 ):
        """
        Grid neighborhood in arbitrary dimension.
        :param type: Type of the neighborhood. So far only linear is supported.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        :param shape: Shape of the topology.
        """
        super().__init__(_Grid(type, size, shape))


class _Grid2D(_Grid):
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: Union[float, int, Callable[..., Union[int, float]]],
                 shape: Tuple[int, int]):
        """
        Grid neighborhood in two dimensions.
        :param type: Type of the neighborhood.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        :param shape: Shape of the topology.
        """
        super().__init__(type, size, shape)

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        Creates random neighborhood and returns it.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        """
        pop_size = len(fitnesses)
        size = self._handle_size(self._size(fitnesses, position, **kwargs), pop_size)
        if pop_size != np.prod(self._shape):
            raise ValueError(f"Grid size doesn't match size of the population, expected {np.prod(self._shape)} individuals")

        if self._type == 'linear':
            return super().__call__(fitnesses, position, **kwargs)
        elif self._type == 'compact':
            orig = np.arange(pop_size)
            indices = np.zeros((pop_size, (2*size+1)**2-1), dtype=int)
            current_index = 0
            width, height = self._shape
            for hi, h in enumerate(range(-size,size+1)):
                for wi, w in enumerate(range(-size,size+1)):
                    if h == 0 and w == 0:
                        continue
                    new_h = (orig // width + height + h) % height
                    new_w = (orig % width + width + w) % width
                    neigh = new_h * width + new_w
                    indices[:,current_index] = neigh
                    current_index += 1
            return t.tensor(indices, dtype=t.long, device=fitnesses.device)
        elif self._type == 'diamond':
            orig = np.arange(pop_size)
            indices = np.zeros((pop_size, size ** 2+(size + 1)**2 - 1), dtype=int)
            current_index = 0
            width, height = self._shape
            for hi, h in enumerate(range(-size,size+1)):
                wmax = size - abs(h)
                for wi, w in enumerate(range(-wmax,wmax+1)):
                    if h == 0 and w == 0:
                        continue
                    new_h = (orig // width + height + h) % height
                    new_w = (orig % width + width + w) % width
                    neigh = new_h * width + new_w
                    indices[:,current_index] = neigh
                    current_index += 1
            return t.tensor(indices, dtype=t.long, device=fitnesses.device)
        else:
            raise ValueError("Invalid mode")


class Grid2D(Static):
    """
    Grid neighborhood in two dimensions.
    """
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: Union[float, int, Callable[..., Union[int, float]]],
                 shape: Tuple[int, int]):
        """
        Grid neighborhood in two dimensions.
        :param type: Type of the neighborhood.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        :param shape: Shape of the topology.
        """
        if len(shape) != 2:
            raise ValueError("Not a 2D grid type")
        super().__init__(_Grid2D(type, size, shape))


class _Circle(_Grid):
    """
    Circle neighborhood.
    """
    def __init__(self, size: Union[float, int, Callable[..., Union[int, float]]]):
        """
        Circle neighborhood.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        """
        super().__init__('linear', size, None)

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        """
        Creates random neighborhood and returns it.
        :param fitnesses: Current particles' fitness.
        :param position: Current particle's positions.
        :param kwargs: Keyword arguments.
        :return: Tensor of indices assigning each particle its neighborhood.
        """
        self._shape = len(fitnesses),
        return super().__call__(fitnesses, position, **kwargs)


class Circle(Static):
    """
    Circle neighborhood.
    """
    def __init__(self, size: Union[float, int, Callable[..., Union[int, float]]]):
        """
        Circle neighborhood.
        :param size: Size of the neighborhood in one direction only. May be float (then it is fraction of the original
        population to select), or integer (then it is number of individuals to select).
        """
        super().__init__(_Circle(size))
