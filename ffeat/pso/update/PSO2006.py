###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Dict, Any, Union, Callable
import math
import torch as t
from .Update import Update

_FDU = Union[float, Callable[..., float], t.distributions.Distribution, Callable[..., t.distributions.Distribution]]


class PSO2006(Update):
    def __init__(self,
                 inertia: _FDU = 1 / (2 * math.log(2)),
                 local_c: _FDU = 0.5 + math.log(2),
                 global_c: _FDU = 0.5 + math.log(2)
                 ):
        self._inertia = self._handle_parameter(inertia)
        self._local_c = self._handle_parameter(local_c)
        self._global_c = self._handle_parameter(global_c)

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[t.Tensor, t.Tensor], Dict[str, Any]]:
        position, velocities, fitness_gbest, positions_gbest, fitness_lbest, positions_lbest = args
        pop_size = len(position)
        ptype = position.dtype
        dev = position.device
        dimensions = len(position.shape[1:])

        inertia = self._inertia(*args, **kwargs)
        local_c = self._local_c(*args, **kwargs)
        if isinstance(local_c, float):
            local_c = t.distributions.Uniform(0.0, local_c)
        global_c = self._global_c(*args, **kwargs)
        if isinstance(global_c, float):
            global_c = t.distributions.Uniform(0.0, global_c)

        if isinstance(inertia, float):
            velocities = t.multiply(t.tensor(inertia, dtype=ptype, device=dev), velocities, out=velocities)
        else:
            inertia = inertia.sample((pop_size,)).to(dev).type(ptype)
            velocities = t.multiply(inertia, velocities, out=velocities)

        local_c = local_c.sample((pop_size,)).to(dev).type(ptype)
        local_shift = positions_lbest - position
        local_shift = t.multiply(
            local_c.reshape(pop_size, *([1] * dimensions)),
            local_shift,
            out=local_shift
        )
        velocities.add_(local_shift)
        del local_c

        global_c = global_c.sample((pop_size,)).to(dev).type(ptype)
        global_shift = t.subtract(positions_gbest, position, out=local_shift)
        global_shift = t.multiply(
            global_c.reshape(pop_size, *([1] * dimensions)),
            global_shift,
            out=global_shift
        )
        velocities.add_(global_shift)
        del global_c

        position = t.add(position, velocities, out=position)

        return (position, velocities), kwargs





