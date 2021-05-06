###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import math


class Exponential:
    """
    Exponential decay rate, parameter decreases by the formula `start * rate ^ iteration`.
    """
    def __init__(self,
                 start: float,
                 min: float = None,
                 rate: float = None,
                 result_type=float):
        """
        Exponential decay rate, parameter decreases by the formula `start * rate ^ iteration`.
        :param start: Start value.
        :param min: Optional minimum value.
        If the algorithm has NOT limited number of iterations, the parameter value will not decrease bellow.
        :param rate: Optional rate. The implementation will multiply the `start` by `rate` every iteration.
        If the algorithm has limited number of iterations and rate is None, it will be automatically calculated so that
        parameter is equal `min` in the last iteration.
        Either `min` or `rate` must be set.
        :param result_type: Type into which the result should by transformed, by default `float`.
        """
        if min is None and rate is None:
            raise ValueError("Either min value or step size must be set")
        self._start = start
        self._min = min
        self._rate = rate
        self._result_type = result_type

    def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs) -> float:
        """
        Calculate new value of the parameter and return it.
        :param args: Arguments.
        :param iteration: Current iteration.
        :param max_iteration: Maximum number of algorithm's iterations, if set.
        :param kwargs: Keyword arguments.
        :return: Exponentially decreasing value of `result_type` type.
        """
        rate = self._rate
        if rate is None and max_iteration is not None and self._min is not None:
            rate = math.pow(self._min / self._start, 1.0 / max_iteration)
        if rate is None:
            raise ValueError("Rate in exponential decay is not known, either set maximum number of iterations or provide rate parameter")
        new_value = self._start * math.pow(rate, iteration)
        if self._min is not None:
            new_value = max(new_value, self._min)
        return self._result_type(new_value)
