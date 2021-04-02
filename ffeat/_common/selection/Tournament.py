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
    def __init__(self,
                 num_select: Union[_IFU, Callable[..., _IFU]] = None,
                 maximization=False,
                 parents: int = 2):
        self.num_select = self._handle_parameter(num_select)
        self.maximization=maximization
        self.parents = parents

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        originally = len(population)
        to_select = self.num_select(fitnesses, population, *args, **kwargs)
        if to_select is None:
            to_select = originally
        if isinstance(to_select, float):
            to_select = int(originally * to_select)
        if not isinstance(to_select, int):
            raise ValueError(f"Number of members to select needs to be int, {type(to_select)} instead")

        indices = t.randint(originally, (to_select, self.parents), dtype=t.long, device=fitnesses.device)
        operation = t.argmax if self.maximization else t.argmin
        best_indices = operation(fitnesses[indices], dim=1)
        selected = population[indices[range(to_select),best_indices]]

        return (selected, *args), kwargs
