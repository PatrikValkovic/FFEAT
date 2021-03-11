###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict, Callable
from ffeat import Normalize


class Lambda(Normalize):
    def __init__(self, _lambda: Callable):
        self._lambda = _lambda

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return super().__call__(self._lambda(*args, **kwargs))
