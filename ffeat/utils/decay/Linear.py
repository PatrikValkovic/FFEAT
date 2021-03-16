###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################

class Linear:
    def __init__(self,
                 start: float,
                 min: float = None,
                 step: float = None,
                 result_type=float):
        if min is None and step is None:
            raise ValueError("Either min value or step size must be set")
        self._start = start
        self._min = min
        self._step = step
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs):
        step = self._step
        if max_iteration is not None and self._min is not None:
            step = (self._start - self._min) / max_iteration
        new_value = self._start - step * iteration
        if self._min is not None:
            new_value = max(new_value, self._min)
        return self._result_type(new_value)
