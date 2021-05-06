###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe
from ffeat.utils._parental_sampling import randint

_IFU = Union[int, float]


class Tournament(Pipe):
    def __init__(self,
                 num_select: Union[_IFU, Callable[..., _IFU]] = None,
                 maximization=False,
                 parents: int = 2,
                 parental_sampling = randint):
        self._num_select = self._handle_parameter(num_select)
        self._maximization=maximization
        self._parents = parents
        self._parental_sampling = parental_sampling

    def __call__(self, fitnesses, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        originally = len(population)
        to_select = self._num_select(fitnesses, population, *args, **kwargs)
        if to_select is None:
            to_select = originally
        if isinstance(to_select, float):
            to_select = int(originally * to_select)
        if not isinstance(to_select, int):
            raise ValueError(f"Number of members to select needs to be int, {type(to_select)} instead")

        indices = self._parental_sampling(originally, to_select, self._parents, fitnesses.device)
        operation = t.argmax if self._maximization else t.argmin
        best_indices = operation(fitnesses[indices], dim=1)
        selected = population[indices[range(to_select),best_indices]]

        return (selected, *args), kwargs
