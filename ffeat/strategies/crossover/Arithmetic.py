###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe


class Arithmetic(Pipe):
    def __init__(self,
                 num_offsprings: int = None,
                 fraction_offsprings: float = None,
                 num_parents: int = 2,
                 parent_weight: Union[float, t.distributions.Distribution] = None,
                 replace_parents: bool = True,
                 in_place: bool = True):
        if num_offsprings is None and fraction_offsprings is None:
            raise ValueError("Either number of offsprings or a percentage must be provided")
        self.num_offsprings = num_offsprings
        self.fraction_offsprings = fraction_offsprings
        self.replace_parents = replace_parents
        self.in_place = in_place
        self.parents = num_parents
        parent_weight = parent_weight or 1 / num_parents
        self.parent_weight = parent_weight if not isinstance(parent_weight, float) else t.distributions.Uniform(parent_weight-1e-6, parent_weight)

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1:]
        num_children = self.num_offsprings if self.num_offsprings is not None else int(len(population) * self.fraction_offsprings)

        parents_indices = t.randint(num_parents, (self.parents * num_children,), dtype=itp, device=dev)
        dist = self.parent_weight
        parents_weights = dist.sample(parents_indices.shape).type(ptp).to(dev)
        parents = population[parents_indices] * parents_weights[:, None]
        children = t.sum(parents.reshape((num_children, self.parents, *dim)), dim=1)

        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices[:num_children]] = children
        else:
            population = t.cat([population, children], dim=0)

        return (population, *args), kwargs
