###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe, flow

_IFU = Union[int, float]


class Elitism(Pipe):
    def __init__(self,
                 num_elites: Union[_IFU, Callable[..., _IFU]],
                 *following_steps: Pipe,
                 maximization=False):
        self._num_elites = self._handle_parameter(num_elites)
        self._maximization = maximization
        self.__follow = flow.Sequence(*following_steps)

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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
