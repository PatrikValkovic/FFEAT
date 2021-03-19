###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe


class UpdatePosition(Pipe):
    def __call__(self, position, velocity, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        position.add_(velocity)
        return (position,), kwargs
