###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class Roulette(Pipe):
    def __init__(self,
                 num_select: Union[_IFU, Callable[..., _IFU]] = None,
                 ):
        self._num_select = self._handle_parameter(num_select)

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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

        normalized = t.divide(fitnesses, t.sum(fitnesses), out=fitnesses)
        cumulative = t.cumsum(normalized, dim=0, out=normalized)
        to_pick = t.rand((to_select), dtype=cumulative.dtype, device=cumulative.device)
        indices = t.sum(cumulative[:, None] < to_pick[None, :], dim=0, dtype=t.long)
        new_population = population[indices.to(population.device)]
        return (new_population, *args), kwargs
