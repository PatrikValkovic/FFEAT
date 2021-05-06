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


class TwoPoint1D(Pipe, _Shared):
    """
    Two point crossover. Works only on one dimensional individuals.
    """

    def __init__(self,
                 offsprings: Union[int, float],
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False,
                 parental_sampling=randint):
        """
        Two point crossover. Works only on one dimensional individuals.
        :param offsprings: Number of offsprings to create, should be dividible by two. May be float (then it is fraction
        of the original population to select), or integer (then it is number of individuals to select).
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
            raise ValueError("Number of offsprings must be even")
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._parental_sampling = parental_sampling

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply two point crossover on the population.
        :param population: Tensor with population. Expect first dimension to enumerate over individuals.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Population with offsprings integrate.
        """
        itp = t.long
        ptp = population.dtype if population.dtype != t.bool else t.uint8
        dev = population.device
        num_parents = len(population)
        dim = population.shape[1]
        num_crossovers = self._offsprings // 2 if isinstance(self._offsprings, int) else int(
            len(population) * self._offsprings / 2.0)
        num_children = num_crossovers * 2

        total = int((dim - 1) * (dim - 2) / 2)
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

        position_mask = t.arange(dim, device=dev).as_strided((num_crossovers, dim), (0, 1))
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
