###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Union, List
from ffeat.Pipe import Pipe, STANDARD_REPRESENTATION


class Select(Pipe):
    """
    Class allowing reordering and selection of the parameters.
    """
    def __init__(self, *selector: Union[int, List[int]]):
        """
        Class allowing reordering and selection of the parameters.
        :param selector: Selector of the parameters. Can be list or variadic number of integers.
        """
        self._selector = selector[0] if len(selector) == 1 and isinstance(selector[0], list) else list(selector)
        self.__params = [None] * len(self._selector)

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Selects arguments and return them, along with unchanged keyword arguments.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: Selected arguments and unchanged keyword arguments.
        """
        for i, selector in enumerate(self._selector):
            if selector >= len(args):
                raise ValueError(f"Attempt to index parameter {selector} but only {len(args)} arguments provided")
            self.__params[i] = args[selector]
        return tuple(self.__params), kwargs
