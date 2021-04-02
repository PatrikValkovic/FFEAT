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
        if isinstance(to_select, int):
            to_select = to_select / originally
        if not isinstance(to_select, float):
            raise ValueError(f"Fraction of elites needs to be float, {type(to_select)} instead")
        if to_select < 0.0 or to_select > 1.0:
            raise ValueError("Fraction of elites must be in range [0.0, 1.0]")
        to_select = to_select if not self._maximization else 1.0 - to_select

        quantile = t.quantile(fitnesses, to_select)
        elites_indices = fitnesses >= quantile if self._maximization else fitnesses <= quantile
        elites = t.clone(population[elites_indices])
        (population, *args), kargs = self.__follow(fitnesses, population, *args, **kwargs)
        if len(population) != len(elites_indices):
            population[t.where(elites_indices)[0]] = elites
        else:
            population[elites_indices] = elites
        return (population, *args), kwargs
