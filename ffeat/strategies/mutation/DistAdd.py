###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


class AddFromDistribution(Pipe):
    def __init__(self,
                 distribution: t.distributions.Distribution,
                 mutation_rate: float = 1.0,
                 in_place: bool = True
                 ):
        if mutation_rate < 0.0 or mutation_rate > 1.0:
            raise ValueError("Mutation rate can't be in the range [0.0, 1.0]")
        self.mutation_rate = mutation_rate
        self.distribution = distribution
        self.in_place = in_place

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        modifications = self.distribution.sample(sample_shape=population.shape).type(population.dtype).to(population.device)
        if self.mutation_rate < 1.0:
            to_modify = (t.rand(population.shape, device=population.device) < self.mutation_rate).type(t.int8)
            modifications = t.multiply(modifications, to_modify, out=modifications)
        population = t.add(population, modifications, out=population if self.in_place else modifications)
        return (population, *args), kwargs


class AddFromNormal(AddFromDistribution):
    def __init__(self, std: float, mutation_rate: float = 1.0, in_place=True):
        super().__init__(t.distributions.Normal(0.0, std), mutation_rate, in_place)
