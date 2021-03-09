###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from ffeat import Pipe

class Parallel(Pipe):
    def __init__(self, *pipes: Pipe):
        self._pipes = pipes

    def __call__(self, *args, **kwargs):
        nargs = []
        nkargs = dict()
        for pipe in self._pipes:
            pargs, pkargs = pipe(*args, **kwargs)
            nargs = nargs + list(pargs)
            nkargs.update(pkargs)
        return nargs, nkargs
