###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class FlipBit(Pipe):
    def __init__(self,
                 num_mutate: Union[_IFU, Callable[..., _IFU]],
                 mutate_prob: Union[float, Callable[..., float]],
                 in_place: bool = True
                 ):
        if isinstance(mutate_prob, float) and (mutate_prob < 0.0 or mutate_prob > 1.0):
            raise ValueError("Mutation probability must be in the range [0.0, 1.0]")
        self._num_mutate = self._handle_parameter(num_mutate)
        self._mutate_prob = self._handle_parameter(mutate_prob)
        self.in_place = in_place

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        pop_size = len(population)
        dim = population.shape[1:]
        dev = population.device
        to_mutate = self._num_mutate(population, *args, **kwargs)
        if isinstance(to_mutate, float):
            to_mutate = int(pop_size * to_mutate)
        mutation_prob = self._mutate_prob(population, *args, **kwargs)
        if mutation_prob < 0.0 or mutation_prob > 1.0:
            raise ValueError("Mutation probability must be in the range [0.0, 1.0]")

        mutation_map = (t.rand((to_mutate,*dim), device=dev) < mutation_prob).to(population.dtype)
        parent_indices = t.randperm(pop_size, device=dev, dtype=t.long)[:to_mutate]
        offsprings = t.subtract(mutation_map, population[parent_indices], out=mutation_map)
        if offsprings.dtype.is_signed:
            offsprings = t.abs(offsprings, out=offsprings)
        else:
            offsprings = t.minimum(offsprings, t.tensor(1), out=offsprings)
        if self.in_place:
            population[parent_indices] = offsprings
        else:
            population = t.cat([population, offsprings], dim=0)

        return (population, *args), kwargs
