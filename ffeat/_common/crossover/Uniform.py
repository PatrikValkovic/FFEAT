###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe
from ._Shared import _Shared


class Uniform(Pipe, _Shared):
    def __init__(self,
                 offsprings: Union[int, float],
                 change_prob: float = 0.5,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        if isinstance(offsprings, int)  and offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even for integer type")
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._change_prob = change_prob

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        ptype = population.dtype
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1:]
        num_crossovers = self._offsprings // 2 if isinstance(self._offsprings, int) else int(len(population) * self._offsprings / 2.0)
        num_children = num_crossovers * 2

        crossover_mask = t.rand((num_crossovers, *dim), device=dev) < self._change_prob
        crossover_mask = crossover_mask.type(t.int8)
        parents_indices = t.randint(num_parents, (2, num_crossovers), dtype=t.long, device=dev)
        children = t.zeros((num_children, *dim), dtype=ptype, device=dev)

        children[:num_crossovers].add_(population[parents_indices[0]] * crossover_mask)
        children[num_crossovers:].add_(population[parents_indices[1]] * crossover_mask)
        crossover_mask = t.logical_not(crossover_mask, out=crossover_mask)
        children[:num_crossovers].add_(population[parents_indices[1]] * crossover_mask)
        children[num_crossovers:].add_(population[parents_indices[0]] * crossover_mask)

        pop = self._handle_pop(population, children, parents_indices)

        return (pop, *args), kwargs
