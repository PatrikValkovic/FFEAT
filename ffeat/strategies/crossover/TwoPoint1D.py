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
class TwoPoint1D(Pipe):
    def __init__(self,
                 num_offsprings: int = None,
                 fraction_offsprings: float = None,
                 replace_parents: bool = True,
                 in_place: bool = True):
        if num_offsprings is None and fraction_offsprings is None:
            raise ValueError("Either number of offsprings or a percentage must be provided")
        if num_offsprings is not None and num_offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even")
        self._num_offsprings = num_offsprings
        self._fraction_offsprings = fraction_offsprings
        self.replace_parents = replace_parents
        self.in_place = in_place

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1]
        num_crossovers = self._num_offsprings // 2 if self._num_offsprings is not None else int(len(population) * self._fraction_offsprings / 2.0)
        num_children = num_crossovers * 2

        first_crossover = t.randint(dim - 2, size=(num_crossovers,), dtype=itp, device=dev) + 1
        second_crossover = t.rand(num_crossovers, device=dev)
        second_crossover = t.multiply(second_crossover, dim - first_crossover - 1, out=second_crossover)
        second_crossover = second_crossover.type(itp).add_(1).add_(first_crossover)
        parents_indices = t.randint(num_parents, (2, num_crossovers), dtype=itp, device=dev)
        children = t.zeros((num_children, dim), dtype=ptp, device=dev)

        position_mask = t.repeat_interleave(t.arange(dim, device=dev, dtype=t.int32)[None,:], repeats=num_crossovers, dim=0)
        mask_first = position_mask >= first_crossover[:, None]
        mask_second = position_mask < second_crossover[:, None]
        mask = t.logical_and(mask_first, mask_second, out=mask_second).type(t.int8)
        children[:num_crossovers].add_(population[parents_indices[0]] * mask)
        children[-num_crossovers:].add_(population[parents_indices[1]] * mask)
        mask = t.logical_not(mask, out=mask)
        children[:num_crossovers].add_(population[parents_indices[1]] * mask)
        children[-num_crossovers:].add_(population[parents_indices[0]] * mask)

        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices.flatten()] = children
        else:
            population = t.cat([population, children], dim=0)

        return (population, *args), kwargs
