###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict

import torch as t
from ffeat._common.initialization.Uniform import UniformInit


class _GAUniform(UniformInit):
    """
    Initialize the population with equal change of gene equal 0 and 1.
    """
    def __init__(self,
                 population_size: int,
                 dimension: int,
                 device: t.device = None,
                 dtype=t.bool):
        """
        Initialize the population with equal change of gene equal 0 and 1.
        :param population_size: Number of individuals in the population.
        :param dimension: Dimensions of each individual.
        :param device: Device on which to allocate the population.
        :param dtype: Torch type by which the population would be represented. By default `torch.bool`.
        """
        super().__init__(population_size, 0, 2, dimension, dtype, device)

    def __call__(self, **kargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """
        Allocated the population and return it.
        :param kargs: Keyword arguments passed along.
        :return: Allocated population with provided keyword arguments.
        """
        (r,), kargs = super().__call__(**kargs)
        r = r.type(self.dtype)
        return (r,), kargs


class Initialization:
    """
    Initialization for Genetic Algorithms.
    """
    Uniform = _GAUniform
