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
    """
    Concatenate results from the pipes to the parameters.
    """
    def __init__(self, *pipes: Pipe):
        """
        Concatenate results from the pipes to the parameters.
        :param pipes: Pipes which results to concatenate.
        """
        super().__init__(Pipe(), *pipes)
