###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ffeat._common.crossover._Shared import _Shared


class Blend(Pipe, _Shared):
    """
    Blend crossover.
    """
    def __init__(self,
                 offsprings: Union[int, float],
                 alpha: Union[float, Callable[..., float]] = 0.5,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        """
        Blend crossover.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param alpha: Alpha parameter, by default 0.5.
        :param replace_parents: Whether should offsprings replace their parents. The operator must create the same
        number of offsprings as there are parents in order for this to work. By default true.
        :param in_place: Whether the new population should be created inplace - in the same memory as the original
        population. The new population size must be equal to the old one. By default true.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._apha = self._handle_parameter(alpha)

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply arithmetic crossover on the population.
        :param population: Tensor with population. Expect first dimension to enumerate over individuals.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Population with offsprings integrate.
        """
        dev = population.device
        dt = population.dtype
        alpha = self._apha(population, *args, **kwargs)
        assert isinstance(alpha, float), 'Alpha is not float'
        num_chilren = self._offsprings if isinstance(self._offsprings, int) else int(self._offsprings * len(population))

        parents = t.randint(len(population), (num_chilren, 2), device=dev, dtype=t.long)
        sub = population[parents[:,1]] - population[parents[:,0]]
        sub.multiply_(alpha)
        lower = population[parents[:,0]] - sub
        upper = t.add(population[parents[:,1]], sub, out=sub)
        mult = upper.subtract_(lower)

        offsprings = t.rand_like(lower) * mult + lower
        pop = self._handle_pop(population, offsprings, parents[:,0])

        return (pop, *args), kwargs

