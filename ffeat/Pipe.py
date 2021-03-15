###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Dict, Any, Callable

class Pipe:
    def _handle_parameter(self, value):
        if isinstance(value, Callable):
            return value
        return lambda *_, **__: value

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return tuple(args), kwargs