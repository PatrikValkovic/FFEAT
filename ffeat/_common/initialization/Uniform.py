###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Tuple, Union, List
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION


class UniformInit(Pipe):
    """
    Initialize the population uniformly in the given interval.
    """
    def __init__(self,
                 population_size: int,
                 min: Union[int, float, List[float], t.Tensor],
                 max: Union[int, float, List[float], t.Tensor],
                 dimension: Union[int, Tuple[int]] = None,
                 dtype: t.dtype = t.float32,
                 device: t.device = None):
        """
        Initialize the population uniformly in the given interval.
        :param population_size: Number of individuals in the population.
        :param min: Minimum value of the gene.
        :param max: Maximum value of the gene.
        :param dimension: Dimensions of each individual.
        :param dtype: Torch type by which the population would be represented. By default `torch.float32`.
        :param device: Device on which to allocate the population.
        """
        self.population_size = population_size
        self.dtype = dtype
        self.device = device
        # handle dimension size
        self.__dimension = dimension   # type: Tuple[int,...]
        if isinstance(self.__dimension, int):
            self.__dimension = (self.__dimension,)
        if self.__dimension is None and isinstance(max, t.Tensor):
            self.__dimension = max.shape
        if self.__dimension is None and isinstance(min, t.Tensor):
            self.__dimension = min.shape
        # handle max and min
        self.__max, self.__min = max, min
        if isinstance(max, int):
            self.__max = float(max)
        if isinstance(min, int):
            self.__min = float(min)
        if isinstance(self.__max, t.Tensor):
            device = self.device = self.__max.device
        if isinstance(self.__min, t.Tensor):
            device = self.device = self.__min.device
        if not isinstance(self.__max, t.Tensor) and isinstance(self.__max, list):
            self.__max = t.tensor(self.__max, device=device)
            self.__dimension = self.__max.shape
        if not isinstance(self.__min, t.Tensor) and isinstance(self.__min, list):
            self.__min = t.tensor(self.__min, device=device)
            self.__dimension = self.__min.shape
        if self.__dimension is None:
            raise ValueError("Dimension not specified, either use dimension parameter or specify dimension of either max or min")
        if not isinstance(self.__max, t.Tensor):
            self.__max = t.full(self.__dimension, self.__max, device=device)
        if not isinstance(self.__min, t.Tensor):
            self.__min = t.full(self.__dimension, self.__min, device=device)
        if self.__max.shape != self.__max.shape or self.__max.shape != self.__dimension or self.__min.shape != self.__dimension:
            raise ValueError("Provided dimensions do not match")
        if t.any(self.__max < self.__min):
            raise ValueError("Maximum can't be lower than minimum")

    def __call__(self, **kargs) -> STANDARD_REPRESENTATION:
        """
        Allocated the population and return it.
        :param kargs: Keyword arguments passed along.
        :return: Allocated population with provided keyword arguments.
        """
        r = t.rand(tuple([self.population_size, *self.__dimension]), device=self.device)
        r = t.multiply(r, self.__max - self.__min, out=r).add_(self.__min)
        r = r.type(self.dtype)
        return (r,), kargs
