###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t
from ffeat import Pipe


class Neighborhood(Pipe):
    def __call__(self, fitnesses, position, **kwargs) -> t.Tensor:
        raise NotImplementedError()
