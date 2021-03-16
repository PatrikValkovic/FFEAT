###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
from typing import Union

_IFU = Union[int, float]

class Linear:
    def __init__(self,
                 start: _IFU,
                 min: _IFU = None,
                 step: _IFU = None,
                 result_type=float):
        if min is None and step is None:
            raise ValueError("Either min value or step size must be set")
        self._start = float(start)
        self._min = float(min) if min is not None else None
        self._step = float(step) if step is not None else None
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs):
        step = self._step
        if max_iteration is not None and self._min is not None:
            step = (self._start - self._min) / max_iteration
        new_value = self._start - step * iteration
        if self._min is not None:
            new_value = max(new_value, self._min)
        return self._result_type(new_value)
