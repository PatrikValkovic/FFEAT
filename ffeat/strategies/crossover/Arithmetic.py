###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe


class Arithmetic(Pipe):
    def __init__(self,
                 num_offsprings: Union[int, Callable[..., int]] = None,
                 fraction_offsprings: Union[float, Callable[..., float]] = None,
                 num_parents: Union[int, Callable[..., int]] = 2,
                 parent_weight: Union[float, t.distributions.Distribution, Callable[..., float], Callable[..., t.distributions.Distribution]] = None,
                 replace_parents: bool = True,
                 in_place: bool = True):
        if num_offsprings is None and fraction_offsprings is None:
            raise ValueError("Either number of offsprings or a percentage must be provided")
        self.num_offsprings = self._handle_parameter(num_offsprings)
        self.fraction_offsprings = self._handle_parameter(fraction_offsprings)
        self.num_parents = self._handle_parameter(num_parents)
        self.replace_parents = replace_parents
        self.in_place = in_place
        self.parent_weight = self._handle_parameter(parent_weight)

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        pop_len = len(population)
        dim = population.shape[1:]
        fract_children = self.fraction_offsprings(population, *args, **kwargs)
        num_children = self.num_offsprings(population, *args, **kwargs)
        num_children = num_children if num_children is not None else int(pop_len * fract_children)
        assert isinstance(num_children, int), f"Number of offsprings should be int, {type(num_children)} received"
        num_parents = self.num_parents(population, *args, **kwargs)
        assert isinstance(num_parents, int), f"Number of parents should be int, {type(num_parents)} received"
        parent_weights = self.parent_weight(population, *args, **kwargs)
        if parent_weights is None:
            parent_weights = 1.0 / num_parents
        if isinstance(parent_weights, float):
            parent_weights = t.distributions.Uniform(parent_weights-1e-7, parent_weights+1e-7)

        parents_indices = t.randint(pop_len, (num_parents * num_children,), dtype=itp, device=dev)
        parents_weights = parent_weights.sample(parents_indices.shape).type(ptp).to(dev)
        parents = population[parents_indices] * t.reshape(parents_weights, (num_parents * num_children, *([1] * len(dim))))
        children = t.sum(parents.reshape((num_children, num_parents, *dim)), dim=1)

        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices[:num_children]] = children
        else:
            population = t.cat([population, children], dim=0)

        return (population, *args), kwargs
