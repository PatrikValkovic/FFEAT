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
                 selection,
                 *following_steps: Pipe):
        self._num_elites = self._handle_parameter(num_elites)
        self.__follow = flow.Sequence(selection, *following_steps)

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        originally = len(population)
        to_select = self._num_elites(fitnesses, population, *args, **kwargs)
        if isinstance(to_select, int):
            to_select = to_select / originally
        if not isinstance(to_select, float):
            raise ValueError(f"Fraction of elites needs to be float, {type(to_select)} instead")
        if to_select < 0.0 or to_select > 1.0:
            raise ValueError("Fraction of elites must be in range [0.0, 1.0]")

        quantile = t.quantile(fitnesses, to_select)
        elites_indices = fitnesses <= quantile
        elites = t.clone(population[elites_indices])
        (population, *args), kargs = self.__follow(fitnesses, population, *args, **kwargs)
        population[elites_indices] = elites
        return (population, *args), kwargs
