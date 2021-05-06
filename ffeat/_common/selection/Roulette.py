###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION

_IFU = Union[int, float]


class Roulette(Pipe):
    """
    Roulette selection operator.
    """
    def __init__(self,
                 num_select: Union[_IFU, Callable[..., _IFU]] = None,
                 ):
        """
        Roulette selection operator. Always expect maximization problem.
        :param num_select: How many individuals to select. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        """
        self._num_select = self._handle_parameter(num_select)

    def __call__(self, fitnesses, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Run the tournament.
        :param fitnesses: Fitness values of the individuals in the same order as they are in the population.
        :param population: Population, where the first dimension enumerate over the individuals.
        :param args: Additional arguments apssed along.
        :param kwargs: Keyword arguments passed along.
        :return: Return the selected population without the fitness functions.
        """
        originally = len(population)
        to_select = self._num_select(fitnesses, population, *args, **kwargs)
        if to_select is None:
            to_select = originally
        if isinstance(to_select, float):
            to_select = int(originally * to_select)
        if not isinstance(to_select, int):
            raise ValueError(f"Number of members to select needs to be int, {type(to_select)} instead")

        if t.any(fitnesses < 0):
            raise ValueError("Fitness with negative values")

        cumulative = t.cumsum(fitnesses, dim=0, out=fitnesses)
        max_val = cumulative[originally-1]
        to_pick = t.rand(to_select, dtype=cumulative.dtype, device=cumulative.device).multiply_(max_val)
        indices = t.sum(cumulative[:, None] < to_pick[None, :], dim=0, dtype=t.long)
        indices = t.minimum(indices, t.tensor(originally-1, device=indices.device, dtype=t.long), out=indices)
        new_population = population[indices]
        return (new_population, *args), kwargs
