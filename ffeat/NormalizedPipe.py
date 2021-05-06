###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict
from .Pipe import Pipe, STANDARD_REPRESENTATION


class NormalizedPipe(Pipe):
    """
    Operator that transform arguments into standard form.
    """

    def __call__(self, argument) -> STANDARD_REPRESENTATION:
        """
        Transform parameter into standard form. Accepts either single value, tuple, or tuple od tuple and dictionary.
        :param argument: Argument to transform.
        :return: Standard parameter form as a tuple of parameters (in tuple), and dictionary.
        """
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
