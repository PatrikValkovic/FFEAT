###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################

from ffeat import Normalize
from typing import Callable

class Lambda(Normalize):
    def __init__(self, _lambda: Callable):
        self._lambda = _lambda

    def __call__(self, *args, **kwargs):
        return super().__call__(self._lambda(*args, **kwargs))
