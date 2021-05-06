###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from ffeat import Pipe, STANDARD_REPRESENTATION


class Parallel(Pipe):
    """
    Executes pipes logically in parallel and join their results.
    """
    def __init__(self, *pipes: Pipe):
        """
        Executes pipes logically in parallel and join their results.
        :param pipes: Pipes to execute in parallel.
        """
        self._pipes = pipes

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Executes the pipes and return their joined results.
        :param args: Arguments. Passed down to all the pipes.
        :param kwargs: Keyword arguments. Passed down to all the pipes.
        :return: Joined results of the pipes.
        """
        nargs = []
        nkargs = dict()
        for pipe in self._pipes:
            pargs, pkargs = pipe(*args, **kwargs)
            nargs = nargs + list(pargs)
            nkargs.update(pkargs)
        return nargs, nkargs
