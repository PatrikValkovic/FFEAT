###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION

_IFU = Union[int, float]


class StochasticUniversalSampling(Pipe):
    """
    Stochastic Universal Sampling selection operator.
    """
    def __init__(self,
                 num_select: Union[_IFU, Callable[..., _IFU]] = None,
                 ):
        """
        Stochastic Universal Sampling selection operator. Always expect maximization problem.
        :param num_select: How many individuals to select. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        """
        self._num_select = self._handle_parameter(num_select)

    def __call__(self, fitnesses, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Run the selection.
        :param fitnesses: Fitness values of the individuals in the same order as they are in the population.
        :param population: Population, where the first dimension enumerate over the individuals.
        :param args: Additional arguments apssed along.
        :param kwargs: Keyword arguments passed along.
        :return: Return the selected population without the fitness functions.
        """
        originally = len(population)
        dev = fitnesses.device
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
        max_val = cumulative[-1]
        to_pick_fitnesses = t.arange(to_select, dtype=t.float32, device=dev)
        to_pick_fitnesses = to_pick_fitnesses.multiply_(
            max_val / to_select
        )
        to_pick_fitnesses.add_(t.rand(1, device=dev, dtype=t.float32) * max_val / to_select)
        indices = t.sum(cumulative[:, None] < to_pick_fitnesses[None, :], dim=0, dtype=t.long)
        indices = t.minimum(indices, t.tensor(originally-1, device=indices.device, dtype=t.long), out=indices)
        new_population = population[indices]
        return (new_population, *args), kwargs
