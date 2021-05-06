###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Optional
from ffeat import Pipe, STANDARD_REPRESENTATION


class Replace(Pipe):
    """
    Allows to replace some arguments by the return of the pipe.
    """
    def __init__(self, pipe: Pipe, param_index: int, num_params: Optional[int] = 1):
        """
        Allows to replace some arguments by the return of the pipe.
        :param pipe: Pipe that accepts all the input arguments and returns the ones, that should replace the old one.
        :param param_index: Index of the first parameter that should be replaced.
        :param num_params: Number of arguments to delete from the original arguments and put the result of the pipe
        in place of them. By default only one arguments will be replaced.
        """
        self._pipe = pipe
        self._param_index = param_index
        self._num_params = num_params

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Call the pipe with input arguments and use its input to replace some of them.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: Replaced arguments with keyword arguments unchanged.
        """
        nparams, nkargs = self._pipe(*args, **kwargs)
        if self._num_params is not None and len(nparams) != self._num_params:
            raise ValueError(f"Returned {len(nparams)} parameters, but expected was {self._num_params}")

        follow_index = self._param_index+(self._num_params if self._num_params is not None else 1)
        to_return_params = [
            *args[:self._param_index],
            *nparams,
            *args[follow_index:]
        ]
        kwargs.update(nkargs)
        return tuple(to_return_params), kwargs
