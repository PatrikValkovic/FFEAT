###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe, flow


class Elitism(Pipe):
    def __init__(self, num_elites: Union[int, float], selection, *following_steps: Pipe):
        if not isinstance(num_elites, float) and not isinstance(num_elites, int):
            raise ValueError("Num elites must be integer or float")
        self.num_elites = num_elites
        self.__follow = flow.Sequence(selection, *following_steps)

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        originally = len(population)
        to_select = self.num_elites
        if isinstance(to_select, int):
            to_select = to_select / originally
        if not isinstance(to_select, float):
            raise ValueError(f"Fraction of elites needs to be float, {type(to_select)} instead")

        quantile = t.quantile(fitnesses, to_select)
        elites_indices = fitnesses <= quantile
        elites = t.clone(population[elites_indices])
        (population, *args), kargs = self.__follow(fitnesses, population, *args, **kwargs)
        population[elites_indices] = elites
        return (population, *args), kwargs
