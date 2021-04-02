###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
from typing import Union, Callable, Tuple, Any, Dict
import torch as t
from ffeat import Pipe
from ffeat._common.crossover._Shared import _Shared


class Blend(Pipe, _Shared):
    def __init__(self,
                 offsprings: Union[int, float],
                 alpha: Union[float, Callable[..., float]] = 0.5,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._apha = self._handle_parameter(alpha)

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        dev = population.device
        dt = population.dtype
        alpha = self._apha(population, *args, **kwargs)
        assert isinstance(alpha, float), 'Alpha is not float'
        num_chilren = self._offsprings if isinstance(self._offsprings, int) else int(self._offsprings * len(population))

        parents = t.randint(len(population), (num_chilren, 2), device=dev, dtype=t.long)
        sub = population[parents[:,1]] - population[parents[:,0]]
        sub.multiply_(alpha)
        lower = population[parents[:,0]] - sub
        upper = t.add(population[parents[:,1]], sub, out=sub)
        mult = upper.subtract_(lower)

        offsprings = t.rand_like(lower) * mult + lower
        pop = self._handle_pop(population, offsprings, parents[:,0])

        return (pop, *args), kwargs

