###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t

from .Neighborhood import Neighborhood


class Static(Neighborhood):
    def __init__(self, neighborhood: Neighborhood):
        self._neighborhood = neighborhood
        self._indices = None

    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        if self._indices is None:
            self._indices = self._neighborhood(fitnesses, position, **kwargs)
        return self._indices
