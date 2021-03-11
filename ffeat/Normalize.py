###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict
from .Pipe import Pipe


class Normalize(Pipe):
    def __call__(self, argument) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if isinstance(argument, tuple) and \
                len(argument) == 2 and \
                isinstance(argument[0], tuple) and \
                isinstance(argument[1], dict):
            return argument
        if isinstance(argument, tuple):
            return argument, {}
        if argument is None:
            return tuple(), {}
        else:
            return (argument,), {}
