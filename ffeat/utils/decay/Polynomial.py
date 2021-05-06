###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import math


class Polynomial:
    """
    Polynomial decay rate, parameter decreases by the formula `start * progress ^ power`.
    """
    def __init__(self,
                 start: float,
                 end: float,
                 power: float,
                 result_type=float):
        """
        Polynomial decay rate, parameter decreases by the formula `start * progress ^ power`.
        :param start: Start value.
        :param end: Minimum value in the end.
        :param power: Power of the polynomial.
        :param result_type: Type into which the result should by transformed, by default `float`.
        """
        self._start = start
        self._end = end
        self._power = power
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int, **kwargs) -> float:
        """
        Calculate new value of the parameter and return it.
        :param args: Arguments.
        :param iteration: Current iteration.
        :param max_iteration: Maximum number of algorithm's iterations. MUST be known.
        :param kwargs: Keyword arguments.
        :return: Polynomial decreasing value of `result_type` type.
        """
        if max_iteration is None:
            raise ValueError("Polynomial decay needs maximum iteration to be known")
        progress = iteration / max_iteration
        decay = math.pow(1.0 - progress, self._power)
        new_rate = (self._start - self._end) * decay + self._end
        return self._result_type(new_rate)
