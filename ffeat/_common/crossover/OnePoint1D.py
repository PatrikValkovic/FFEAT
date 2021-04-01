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


# TODO what to do with multiple dimensions
class OnePoint1D(Pipe, _Shared):
    def __init__(self,
                 offsprings: Union[int, float],
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        if isinstance(offsprings, int) and offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even")
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1]
        num_crossovers = self._offsprings // 2 if isinstance(self._offsprings, int) else int(len(population) * self._offsprings / 2.0)
        num_children = num_crossovers * 2

        crossover_indices = t.randint(dim - 1, size=(num_crossovers,), dtype=itp, device=dev) + 1
        parents_indices = t.randint(num_parents, (2, num_crossovers), dtype=itp, device=dev)
        children = t.zeros((num_children, dim), dtype=ptp, device=dev)

        position_mask = t.repeat_interleave(t.arange(dim, device=dev)[None,:], repeats=num_crossovers, dim=0)
        lpos = (position_mask < crossover_indices[:, None]).type(t.int8)
        children[:num_crossovers].add_(population[parents_indices[0]] * lpos)
        children[num_crossovers:].add_(population[parents_indices[1]] * lpos)
        rpos = t.logical_not(lpos, out=lpos)
        children[:num_crossovers].add_(population[parents_indices[1]] * rpos)
        children[num_crossovers:].add_(population[parents_indices[0]] * lpos)

        pop = self._handle_pop(population, children, parents_indices)

        return (pop, *args), kwargs
