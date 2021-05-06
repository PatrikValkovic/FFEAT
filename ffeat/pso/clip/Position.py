###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe, STANDARD_REPRESENTATION


class Position(Pipe):
    """
    Clip particles' position into given range.
    """
    def __init__(self, min, max):
        """
        Clip particles' position into given range.
        :param min: Minimum position in each dimension.
        :param max: Maximum position in each dimension.
        """
        self._min = min
        self._max = max

    def __call__(self, position, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Clips positions and return them.
        :param position: Current positions.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Update positions with given arguments.
        """
        position.clip_(self._min, self._max)
        return (position, *args), kwargs
