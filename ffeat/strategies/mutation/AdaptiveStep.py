###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
from typing import Union
import math
import torch as t
import ffeat


class AdaptiveStep(ffeat.Pipe):
    """
    Normal mutation with adaptive step.
    """
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
        """
        Normal mutation with adaptive step.
        :param init_std: Starting standard deviation.
        :param std_increase: Increase step of standard deviation. The original deviation is multiplied by this value.
        :param evaluation: Evaluation pipe to evaluate new offsprings.
        :param std_decrease: Decrease step of standard deviation. The original deviation is multiplied by this value.
        If not provided uses 1 / `std_increase` value.
        :param mutate_members: How many individuals to mutate. Default mutate whole population.
        May be float (then it is fraction of the original population to select), or integer (then it is number of individuals to select).
        :param better_to_increase: Minimum number of better offsprings to increase the standard deviation. Default 0.2.
        :param replace_parents: Whether should offsprings replace their children (default) or concatenate themselves to the population.
        :param replace_only_better: Whether to replace parent only if offsprings is better. Default false.
        :param minimum_std: Maximum value of the standard deviation. By default unlimited.
        :param maximum_std: Maximum value of the standard deviation. By default unlimited.
        """
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

    def __call__(self, fitnesses, population, *args, **kwargs) -> ffeat.STANDARD_REPRESENTATION:
        """
        Mutate the population and return it.
        :param fitnesses: Fitness of the population. Expect to be one dimensional array
        in the same order as the population.
        :param population: Tensor representing the population.
        Expects the first dimension enumerate over the individuals.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Return fitness and mutated population. Moreover add `better_fraction` keyword argument (fraction of
        better offsprings) and `current_std` keyword argument (new normal's mutation standard deviation).
        """
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
        kwargs['better_fraction'] = num_better / num_parents
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
