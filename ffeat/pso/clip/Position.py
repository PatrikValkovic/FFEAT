###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe


class Position(Pipe):
    def __init__(self, min, max):
        self._min = min
        self._max = max

    def __call__(self, position, velocities, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        position.clip_(self._min, self._max)
        return (position, velocities), kwargs
