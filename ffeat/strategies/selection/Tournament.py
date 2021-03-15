###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]

class Tournament(Pipe):
    def __init__(self, num_select: Union[_IFU, Callable[..., _IFU]] = None):
        self.num_select = self._handle_parameter(num_select)

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        originally = len(population)
        to_select = self.num_select(fitnesses, population, *args, **kwargs)
        if to_select is None:
            to_select = originally
        if isinstance(to_select, float):
            to_select = int(originally * to_select)
        if not isinstance(to_select, int):
            raise ValueError(f"Number of members to select needs to be int, {type(to_select)} instead")

        indices = t.randint(originally, (2, to_select), dtype=t.long, device=population.device)
        comparison = fitnesses[indices[0]] < fitnesses[indices[1]]
        better = t.cat([
            indices[0, comparison],
            indices[1, t.logical_not(comparison)]
        ])
        new_population = population[better]
        return (new_population, *args), kwargs
