###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe, STANDARD_REPRESENTATION


class Sequence(Pipe):
    """
    Run pipes logically in sequence.
    """
    def __init__(self, *steps: Pipe):
        """
        Run pipes logically in sequence.
        :param steps: Pipes to call in sequence.
        """
        self.__steps = steps

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Run the pipes in sequence and return output of the last one.
        :param args: Arguments passed into the first pipe.
        :param kwargs: Keyword arguments passed into the first pipe.
        :return: Output of the last pipe.
        """
        for step in self.__steps:
            args, kwargs = step(*args, **kwargs)
        return args, kwargs
