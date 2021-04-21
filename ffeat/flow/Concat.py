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
        super().__init__(Pipe(), *pipes)
