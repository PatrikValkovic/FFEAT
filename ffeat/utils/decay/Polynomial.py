###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import math


class Polynomial:
    def __init__(self,
                 start: float,
                 end: float,
                 power: float,
                 result_type=float):
        self._start = start
        self._end = end
        self._power = power
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int, **kwargs) -> float:
        if max_iteration is None:
            raise ValueError("Polynomial decay needs maximum iteration to be known")
        progress = iteration / max_iteration
        decay = math.pow(1.0 - progress, self._power)
        new_rate = (self._start - self._end) * decay + self._end
        return self._result_type(new_rate)
