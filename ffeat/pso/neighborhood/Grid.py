###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Literal
import torch as t
import numpy as np
from .Neighborhood import Neighborhood
from .Static import Static


class _Grid(Neighborhood):
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: int,
                 shape: Tuple[int, ...]):
        self._type = type
        self._size = size
        self._shape = shape

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        pop_size = len(fitnesses)
        dimensions = len(self._shape)
        if pop_size != np.prod(self._shape):
            raise ValueError(f"Grid size doesn't match size of the population, expected {np.prod(self._shape)} individuals")

        orig = np.arange(pop_size)
        if self._type == 'linear':
            indices = np.zeros((pop_size, 2*self._size*dimensions), dtype=int)
            for dim_i, dim in enumerate(self._shape):
                div_back = np.prod(self._shape[:dim_i])
                div_forw = np.prod(self._shape[:dim_i+1])
                for step in range(self._size):
                    pos_in_dim = (orig // div_back) % div_forw
                    add_forward = orig // div_forw * div_forw
                    add_backward = orig % div_back
                    neigh_pos_in_dim = (pos_in_dim + step + 1) % dim
                    neigh = neigh_pos_in_dim * div_back + add_forward + add_backward
                    indices[:,step+self._size*dim_i*2] = neigh
                for step in range(self._size):
                    pos_in_dim = (orig // div_back) % div_forw
                    add_forward = orig // div_forw * div_forw
                    add_backward = orig % div_back
                    neigh_pos_in_dim = (pos_in_dim + dim - 1 - step) % dim
                    neigh = neigh_pos_in_dim * div_back + add_forward + add_backward
                    indices[:,step+self._size+self._size*dim_i*2] = neigh
            return t.tensor(indices, dtype=t.long, device=fitnesses.device)
        # TODO compact and diamond shape
        raise NotImplementedError()


class Grid(Static):
    def __init__(self,
                 type: Literal["linear", "compact", "diamond"],
                 size: int,
                 shape: Tuple[int, ...]
                 ):
        super().__init__(_Grid(type, size, shape))


class _Grid2D(_Grid):
    def __init__(self, type: Literal["linear", "compact", "diamond"], size: int, shape: Tuple[int, ...]):
        super().__init__(type, size, shape)

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        pop_size = len(fitnesses)
        if pop_size != np.prod(self._shape):
            raise ValueError(f"Grid size doesn't match size of the population, expected {np.prod(self._shape)} individuals")

        if self._type == 'linear':
            return super().__call__(fitnesses, position, **kwargs)
        elif self._type == 'compact':
            orig = np.arange(pop_size)
            indices = np.zeros((pop_size, (2*self._size+1)**2-1), dtype=int)
            current_index = 0
            width, height = self._shape
            for hi, h in enumerate(range(-self._size,self._size+1)):
                for wi, w in enumerate(range(-self._size,self._size+1)):
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
            indices = np.zeros((pop_size, self._size ** 2+(self._size + 1)**2 - 1), dtype=int)
            current_index = 0
            width, height = self._shape
            for hi, h in enumerate(range(-self._size,self._size+1)):
                wmax = self._size - abs(h)
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
    def __init__(self, type: Literal["linear", "compact", "diamond"], size: int, shape: Tuple[int, ...]):
        if len(shape) != 2:
            raise ValueError("Not a 2D grid type")
        super().__init__(_Grid2D(type, size, shape))


class _Circle(_Grid):
    def __init__(self, size: int):
        super().__init__('linear', size, None)

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        self._shape = len(fitnesses),
        return super().__call__(fitnesses, position, **kwargs)


class Circle(Static):
    def __init__(self, size: int):
        super().__init__(_Circle(size))
