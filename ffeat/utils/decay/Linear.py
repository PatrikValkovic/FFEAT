###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
from typing import Union

_IFU = Union[int, float]

class Linear:
    """
    Linear decay rate, parameter decreases linearly from `start` to `min`.
    """
    def __init__(self,
                 start: _IFU,
                 min: _IFU = None,
                 step: _IFU = None,
                 result_type=float):
        """
        Linear decay rate, parameter decreases linearly from `start` to `min`.
        :param start: Start value.
        :param min: Optional minimum value.
        If the algorithm has NOT limited number of iterations, the parameter value will not decrease bellow.
        :param step: Optional step size. The implementation will decrease the parameter by this value every iteration.
        If the algorithm has limited number of iterations and step is None, it will be automatically calculated so that
        parameter is equal `min` in the last iteration.
        Either `min` or `step` must be set.
        :param result_type: Type into which the result should by transformed, by default `float`.
        """
        if min is None and step is None:
            raise ValueError("Either min value or step size must be set")
        self._start = float(start)
        self._min = float(min) if min is not None else None
        self._step = float(step) if step is not None else None
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs):
        """
        Calculate new value of the parameter and return it.
        :param args: Arguments.
        :param iteration: Current iteration.
        :param max_iteration: Maximum number of algorithm's iterations, if set.
        :param kwargs: Keyword arguments.
        :return: Linearly decreasing value of `result_type` type.
        """
        step = self._step
        if step is None and max_iteration is not None and self._min is not None:
            step = (self._start - self._min) / max_iteration
        if step is None:
            raise ValueError("Step in linear decay is not known, either set maximum number of iterations or provide step parameter")
        new_value = self._start - step * iteration
        if self._min is not None:
            new_value = max(new_value, self._min)
        return self._result_type(new_value)
