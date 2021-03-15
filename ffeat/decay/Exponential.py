###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import math

class Exponential:
    def __init__(self,
                 start: float,
                 min: float = None,
                 rate: float = None,
                 result_type=float):
        if min is None and rate is None:
            raise ValueError("Either min value or step size must be set")
        self._start = start
        self._min = min
        self._rate = rate
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs) -> float:
        rate = self._rate
        if max_iteration is not None and self._min is not None:
            rate = math.pow(self._min / self._start, 1.0 / max_iteration)
        new_value = self._start * math.pow(rate, iteration)
        if self._min is not None:
            new_value = max(new_value, self._min)
        return self._result_type(new_value)
