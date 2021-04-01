###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe
from ffeat._common.crossover._Shared import _Shared


class Arithmetic(Pipe, _Shared):
    def __init__(self,
                 offsprings: Union[int, float],
                 num_parents: Union[int, Callable[..., int]] = 2,
                 parent_weight: Union[float, t.distributions.Distribution, Callable[..., float], Callable[..., t.distributions.Distribution]] = None,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self.num_parents = self._handle_parameter(num_parents)
        self.parent_weight = self._handle_parameter(parent_weight)

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        pop_len = len(population)
        dim = population.shape[1:]
        num_children = self._offsprings
        num_children = num_children if isinstance(num_children, int) else int(pop_len * num_children)
        assert isinstance(num_children, int), f"Number of offsprings should be int, {type(num_children)} received"
        num_parents = self.num_parents(population, *args, **kwargs)
        assert isinstance(num_parents, int), f"Number of parents should be int, {type(num_parents)} received"
        parent_weights = self.parent_weight(population, *args, **kwargs)
        if parent_weights is None:
            parent_weights = 1.0 / num_parents
        if isinstance(parent_weights, float):
            parent_weights = t.distributions.Uniform(parent_weights-1e-7, parent_weights+1e-7)

        # TODO make sure weights corresponds
        parents_indices = t.randint(pop_len, (num_parents * num_children,), dtype=itp, device=dev)
        parents_weights = parent_weights.sample((num_children, num_parents)).type(ptp).to(dev)
        parents = population[parents_indices] * t.reshape(parents_weights, (num_parents * num_children, *([1] * len(dim))))
        children = t.sum(parents.reshape((num_children, num_parents, *dim)), dim=1)

        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices[:num_children]] = children
        else:
            population = t.cat([population, children], dim=0)

        return (population, *args), kwargs
