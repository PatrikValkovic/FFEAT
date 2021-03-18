###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


class UpdateLocalBest(Pipe):
    def __call__(self, fitnesses, position, fitness_lbest, positions_lbest, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        better = fitnesses < fitness_lbest
        fitness_lbest = t.minimum(fitnesses, fitness_lbest, out=fitness_lbest)
        positions_lbest[better] = position[better]
        return (fitness_lbest, positions_lbest), {}
