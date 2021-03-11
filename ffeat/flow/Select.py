###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Union, List, Tuple, Any, Dict
from ffeat.Pipe import Pipe


class Select(Pipe):
    def __init__(self, *selector: Union[int, List[int]]):
        self._selector = selector[0] if len(selector) == 1 and isinstance(selector[0], list) else list(selector)
        self.__params = [None] * len(self._selector)

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        for i, selector in enumerate(self._selector):
            if selector >= len(args):
                raise ValueError(f"Attempt to index parameter {selector} but only {len(args)} arguments provided")
            self.__params[i] = args[selector]
        return tuple(self.__params), kwargs
