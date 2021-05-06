###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, flow, STANDARD_REPRESENTATION

_IFU = Union[int, float]


class Elitism(Pipe):
    """
    Elitism operator. Copy elites from the population, provided pipes and then copies them back.
    """
    def __init__(self,
                 num_elites: Union[_IFU, Callable[..., _IFU]],
                 *following_steps: Pipe,
                 maximization=False):
        """
        Elitism operator. Copy elites from the population, provided pipes and then copies them back.
        :param num_elites: Number of elites. May be float (then it is fraction of the original population to select),
        or integer (then it is number of individuals to select).
        :param following_steps: Rest of the steps in the algorithm.
        Expect of steps to return population without the fitness.
        :param maximization: Whether it is maximization or minimization (default) problem.
        """
        self._num_elites = self._handle_parameter(num_elites)
        self._maximization = maximization
        self.__follow = flow.Sequence(*following_steps)

    def __call__(self, fitnesses, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Copy elites from the population, provided pipes and then copies them back.
        :param fitnesses: Fitness values of the individuals in the same order as they are in the population.
        :param population: Population, where the first dimension enumerate over the individuals.
        :param args: Arguments to passed along.
        :param kwargs: Keyword arguments to passed along.
        :return: Population after rest of the operators applied and elites copied back into the population.
        """
        originally = len(population)
        to_select = self._num_elites(fitnesses, population, *args, **kwargs)
        if isinstance(to_select, float):
            to_select = int(to_select * originally)
        if not isinstance(to_select, int):
            raise ValueError(f"Number of elites needs to be int, {type(to_select)} instead")
        if to_select > originally:
            raise ValueError(f"Number attempt to get {to_select} elites for population of size {originally}")

        elites_indices = t.topk(fitnesses, to_select, largest=self._maximization)[1]
        elites = t.clone(population[elites_indices])
        (population, *args), kargs = self.__follow(fitnesses, population, *args, **kwargs)
        population[elites_indices] = elites
        return (population, *args), kwargs
