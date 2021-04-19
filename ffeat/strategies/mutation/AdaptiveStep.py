###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import math
import torch as t
import ffeat


class AdaptiveStep(ffeat.Pipe):
    def __init__(self,
                 init_std: float,
                 std_increase: float,
                 evaluation,
                 std_decrease: float = None,
                 mutate_members: Union[float, int] = 1.0,
                 better_to_increase: Union[float, int] = 0.2,
                 replace_parents: bool = True,
                 replace_only_better: bool = False,
                 minimum_std: float = None,
                 maximum_std: float = None):
        self._current_std = init_std
        self._std_increase = std_increase
        self._std_decrease = std_decrease or 1 / self._std_increase
        self._evaluation = evaluation
        self._replace_parents = replace_parents
        self._replace_only_better = replace_only_better
        self._maximum_std = maximum_std or math.inf
        self._minimum_std = minimum_std or 1e-12
        self._mutate_members = mutate_members
        self._better_to_increase = better_to_increase

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        dev, ptype = population.device, population.type()
        pop_size = len(population)
        dim = population.shape[1:]
        num_parents = self._mutate_members if isinstance(self._mutate_members, int) else int(pop_size * self._mutate_members)
        needs_better = self._better_to_increase if isinstance(self._better_to_increase, int) else int(pop_size * self._better_to_increase)
        dist = t.distributions.Normal(0.0, self._current_std)

        if num_parents == pop_size:
            parent_indices = t.arange(pop_size, dtype=t.long, device=dev)
        else:
            parent_indices = t.randint(pop_size, (num_parents,), dtype=t.long, device=dev)
        mutation = dist.sample((num_parents, *dim)).type(ptype).to(dev)
        offsprings = population[parent_indices] + mutation
        (ofitnesses, offsprings), kargs = self._evaluation(offsprings, *args, **kwargs)

        better_offsprings = ofitnesses < fitnesses[parent_indices]
        num_better = t.sum(better_offsprings)
        kwargs['better_fraction'] = num_better / pop_size
        if num_better >= needs_better:
            self._current_std *= self._std_increase
        else:
            self._current_std *= self._std_decrease
        self._current_std = min(self._maximum_std, max(self._minimum_std, self._current_std))
        kwargs['current_std'] = self._current_std

        if self._replace_only_better:
            population[parent_indices[better_offsprings]] = offsprings[better_offsprings]
            fitnesses[parent_indices[better_offsprings]] = ofitnesses[better_offsprings]
        elif self._replace_parents:
            population[parent_indices] = offsprings
            fitnesses[parent_indices] = ofitnesses
        else:
            population = t.cat([population, offsprings], dim=0)
            fitnesses = t.cat([fitnesses, ofitnesses], dim=0)

        return (fitnesses, population, *args), kwargs
