###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Union
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ._Shared import _Shared
from ffeat.utils._parental_sampling import randint


class Uniform(Pipe, _Shared):
    """
    Uniform crossover operator.
    """

    def __init__(self,
                 offsprings: Union[int, float],
                 change_prob: float = 0.5,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False,
                 parental_sampling=randint):
        """
        Uniform crossover oeprator.
        :param offsprings: Number of offsprings to create, should be dividible by two. May be float (then it is fraction
        of the original population to select), or integer (then it is number of individuals to select).
        :param change_prob: Probability to inherit gene from the first parent.
        :param replace_parents: Whether should offsprings replace their parents. The operator must create the same
        number of offsprings as there are parents in order for this to work. By default true.
        :param in_place: Whether the new population should be created inplace - in the same memory as the original
        population. The new population size must be equal to the old one. By default true.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        :param parental_sampling: How should be individuals for the single tournament sampled.
        By default uses `randint`.
        """
        if isinstance(offsprings, int) and offsprings % 2 != 0:
            raise ValueError("Number of offsprings must be even for integer type")
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._change_prob = change_prob
        self._parental_sampling = parental_sampling

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply uniform crossover on the population.
        :param population: Tensor with population. Expect first dimension to enumerate over individuals.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Population with offsprings integrate.
        """
        ptype = population.dtype if population.dtype != t.bool else t.uint8
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1:]
        num_crossovers = self._offsprings // 2 if isinstance(self._offsprings, int) else int(
            len(population) * self._offsprings / 2.0)
        num_children = num_crossovers * 2

        crossover_mask = t.rand((num_crossovers, *dim), device=dev) < self._change_prob
        crossover_mask = crossover_mask.type(t.int8)
        parents_indices = self._parental_sampling(num_parents, num_crossovers, 2, dev).T
        children = t.zeros((num_children, *dim), dtype=ptype, device=dev)

        children[:num_crossovers].add_(population[parents_indices[0]] * crossover_mask)
        children[num_crossovers:].add_(population[parents_indices[1]] * crossover_mask)
        crossover_mask = t.logical_not(crossover_mask, out=crossover_mask)
        children[:num_crossovers].add_(population[parents_indices[1]] * crossover_mask)
        children[num_crossovers:].add_(population[parents_indices[0]] * crossover_mask)
        children = children.to(population.dtype)

        pop = self._handle_pop(population, children, parents_indices)

        return (pop, *args), kwargs
