###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Dict, Any, Callable

STANDARD_REPRESENTATION = Tuple[Tuple[Any, ...], Dict[str, Any]]
"""
Standard representation of the arguments in the library.
"""

class Pipe:
    """
    Base class for all the pipes and operators in the library.
    """

    def _handle_parameter(self, value):
        """
        Helper method that transform value into function that returns it, or return the original callable object otherwise.
        :param value: Value to transform.
        :return: Callable object with variadic number of parameters.
        """
        if isinstance(value, Callable):
            return value
        return lambda *_, **__: value

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Base operator the devrived class should overwrite. By default returns its parameters.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: Tuple of parameters as tuple, and keyword arguments as dictionary.
        """
        return tuple(args), kwargs
