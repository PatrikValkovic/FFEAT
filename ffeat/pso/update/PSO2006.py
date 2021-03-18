###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Dict, Any, Union, Callable
import torch as t
from .Update import Update

_FDU = Union[float, Callable[..., float], t.distributions.Distribution, Callable[..., t.distributions.Distribution]]


class PSO2006(Update):
    def __init__(self,
                 inertia: _FDU = 0.7,
                 local_c: _FDU = 1.9,
                 global_c: _FDU = 2.0
                 ):
        self._inertia = self._handle_parameter(inertia)
        self._local_c = self._handle_parameter(local_c)
        self._global_c = self._handle_parameter(global_c)

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[t.Tensor, t.Tensor], Dict[str, Any]]:
        position, velocities, fitness_gbest, positions_gbest, fitness_lbest, positions_lbest = args
        pop_size = len(position)
        ptype = position.dtype
        dev = position.device

        inertia = self._inertia(*args, **kwargs)
        local_c = self._local_c(*args, **kwargs)
        if isinstance(local_c, float):
            local_c = t.distributions.Uniform(0.0, local_c)
        global_c = self._global_c(*args, **kwargs)
        if isinstance(global_c, float):
            global_c = t.distributions.Uniform(0.0, global_c)

        local_c = local_c.sample((pop_size,)).to(dev).type(ptype)
        local_shift = positions_lbest - position
        local_shift = t.multiply(local_c, local_shift, out=local_shift)
        velocities.add_(local_shift)
        del local_c

        global_c = global_c.sample((pop_size,)).to(dev).type(ptype)
        local_shift = t.subtract(positions_gbest, position, out=local_shift)
        local_shift = t.multiply(global_c, local_shift, out=local_shift)
        velocities.add_(local_shift)
        del global_c

        if isinstance(inertia, float):
            velocities = t.multiply(t.tensor(inertia, dtype=ptype, device=dev), velocities, out=velocities)
        else:
            inertia = inertia.sample((pop_size,)).to(dev).type(ptype)
            velocities = t.multiply(inertia, velocities, out=velocities)

        position = t.add(position, velocities, out=position)

        return (position, velocities), kwargs





