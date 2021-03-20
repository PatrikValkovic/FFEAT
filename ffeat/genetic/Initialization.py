###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Tuple, Any, Dict

import torch as t
from ffeat._common.initialization.Uniform import UniformInit


class _GAUniform(UniformInit):
    def __init__(self,
                 population_size: int,
                dimension: int,
                device: t.device = None):
        super().__init__(population_size, 0, 2, dimension, t.int8, device)

    def __call__(self, **kargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        (r,), kargs = super().__call__(**kargs)
        r = r.type(t.bool)
        return (r,), kargs


class Initialization:
    Uniform = _GAUniform
