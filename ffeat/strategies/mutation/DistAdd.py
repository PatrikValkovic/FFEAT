###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


# TODO tests
class AddFromDistribution(Pipe):
    def __init__(self,
                 distribution: t.distributions.Distribution,
                 mutation_rate: float = 1.0
                 ):
        if mutation_rate < 0.0 or mutation_rate > 1.0:
            raise ValueError("Mutation rate can't be in the range [0.0, 1.0]")
        self.mutation_rate = mutation_rate
        self.distribution = distribution

    def __call__(self, population, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        modifications = self.distribution.sample(sample_shape=population.shape).type(population.dtype).to(population.device)
        if self.mutation_rate < 1.0:
            to_modify = (t.rand(population.shape, device=population.device) < self.mutation_rate).type(t.int8)
            modifications = t.multiply(modifications, to_modify, out=modifications)
        population.add_(modifications)
        return (population,), kwargs


class AddFromNormal(AddFromDistribution):
    def __init__(self, std: float, mutation_rate: float = 1.0):
        super().__init__(t.distributions.Normal(0.0, std), mutation_rate)
