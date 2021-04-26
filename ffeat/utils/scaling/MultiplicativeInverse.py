###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


class MultiplicativeInverse(Pipe):
    def __call__(self, fitnesses, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        fitnesses = t.divide(t.tensor(1.0), fitnesses, out=fitnesses if fitnesses.dtype.is_floating_point else None)
        kwargs['new_fitness'] = fitnesses
        return (fitnesses, *args), kwargs