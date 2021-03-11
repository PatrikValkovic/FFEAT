###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe
from . import Parallel


class Concat(Parallel):
    def __init__(self, *pipes: Pipe):
        super().__init__(*pipes)

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        nargs, nkargs = super().__call__(*args, **kwargs)
        kwargs.update(nkargs)
        return tuple(list(args) + list(nargs)), kwargs
