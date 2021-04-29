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
from ffeat.utils._parental_sampling import randint


# TODO what to do with multiple dimensions
class TwoPoint1D(Pipe, _Shared):
    def __init__(self,
                 offsprings: Union[int, float],
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False,
                 parental_sampling = randint):
        if isinstance(offsprings, int) and offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even")
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._parental_sampling = parental_sampling

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype if population.dtype != t.bool else t.uint8
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1]
        num_crossovers = self._offsprings // 2 if isinstance(self._offsprings, int) else int(len(population) * self._offsprings / 2.0)
        num_children = num_crossovers * 2

        total = int((dim-1) * (dim - 2) / 2)
        first_crossover = t.rand(num_crossovers, device=dev)
        b = 2 * (dim - 2) + 1
        D = t.subtract(
            t.tensor(b ** 2, dtype=first_crossover.dtype, device=dev),
            first_crossover,
            alpha=total * 8,
            out=first_crossover
        )
        D.sqrt_()
        first_crossover = t.subtract(
            t.tensor(b / 2, dtype=first_crossover.dtype, device=dev),
            first_crossover,
            alpha=0.5,
            out=D
        ).type(t.long)
        first_crossover.add_(1)

        second_crossover = t.rand(num_crossovers, device=dev)
        second_crossover = t.multiply(second_crossover, dim - first_crossover - 1, out=second_crossover)
        second_crossover = second_crossover.add_(1).add_(first_crossover).type(itp)
        parents_indices = self._parental_sampling(num_parents, num_crossovers, 2, dev).T
        children = t.zeros((num_children, dim), dtype=ptp, device=dev)

        position_mask = t.arange(dim, device=dev).as_strided((num_crossovers,dim), (0,1))
        mask_first = position_mask >= first_crossover[:, None]
        mask_second = position_mask < second_crossover[:, None]
        mask = t.logical_and(mask_first, mask_second, out=mask_second).type(t.int8)
        children[:num_crossovers].add_(population[parents_indices[0]] * mask)
        children[-num_crossovers:].add_(population[parents_indices[1]] * mask)
        mask = t.logical_not(mask, out=mask)
        children[:num_crossovers].add_(population[parents_indices[1]] * mask)
        children[-num_crossovers:].add_(population[parents_indices[0]] * mask)
        children = children.to(population.dtype)

        pop = self._handle_pop(population, children, parents_indices)

        return (pop, *args), kwargs
