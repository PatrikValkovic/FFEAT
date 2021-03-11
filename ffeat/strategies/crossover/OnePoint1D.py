###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


# TODO what to do with multiple dimensions
class OnePoint1D(Pipe):
    def __init__(self,
                 num_offsprings: int = None,
                 crossover_percentage: float = None,
                 replace_parents: bool = True):
        if num_offsprings is None and crossover_percentage is None:
            raise ValueError("Either number of offsprings or a percentage must be provided")
        if num_offsprings is not None and num_offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even")
        self.num_offsprings = num_offsprings
        self.crossover_percentage = crossover_percentage
        self.replace_parents = replace_parents

    def __call__(self, population, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1]
        num_crossovers = self.num_offsprings // 2 if self.num_offsprings is not None else int(len(population) * self.crossover_percentage / 2.0)
        num_children = num_crossovers * 2

        crossover_indices = t.randint(dim - 1, size=(num_crossovers,), dtype=itp, device=dev) + 1
        parents_indices = t.randint(num_parents, (2, num_crossovers), dtype=itp, device=dev)
        children = t.zeros((num_children, dim), dtype=ptp, device=dev)

        position_mask = t.repeat_interleave(t.arange(dim, device=dev)[None,:], repeats=num_crossovers, dim=0)
        lpos = (position_mask < crossover_indices[:, None]).type(t.int8)
        children[:num_crossovers].add_(population[parents_indices[0]] * lpos)
        children[-num_crossovers:].add_(population[parents_indices[1]] * lpos)
        rpos = t.logical_not(lpos, out=lpos)
        children[:num_crossovers].add_(population[parents_indices[1]] * rpos)
        children[-num_crossovers:].add_(population[parents_indices[0]] * lpos)

        if self.replace_parents:
            population[parents_indices.flatten()] = children
        else:
            population = t.cat([population, children], dim=0)

        return (population,), kwargs
