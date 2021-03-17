###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
import torch as t
from ffeat._common.initialization.Uniform import UniformInit


class Initialization:
    @staticmethod
    def Uniform(population_size: int,
                dimension: int,
                device: t.device = None) -> UniformInit:
        return UniformInit(population_size, 0, 2, dimension, t.uint8, device)
