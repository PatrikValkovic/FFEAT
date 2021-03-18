###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe


class Update(Pipe):
    def __call__(self, position, velocities,
                 fitness_gbest, positions_gbest,
                 fitness_lbest, positions_lbest, **kwargs) -> Tuple[Tuple[t.Tensor, t.Tensor], Dict[str, Any]]:
        raise NotImplementedError()
