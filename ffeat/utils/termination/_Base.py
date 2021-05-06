###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Tuple, Any, Dict
import numpy as np
import ffeat


class _Base(ffeat.Pipe):
    """
    Base class for early termination logic.
    """

    def __init__(self, for_steps: int):
        """
        Base class for early termination logic.
        :param for_steps: For how many steps to measure.
        """
        self._steps = for_steps
        self._current_index = 0
        self._measured = 0
        self._values = np.zeros(self._steps, dtype=float)

    def _log(self, value):
        """
        Log value in current iteration. Store it into the buffer.
        :param value: Value to log.
        :return: None
        """
        self._values[self._current_index] = float(value)
        self._measured = min(self._measured + 1, self._steps)
        self._current_index = (self._current_index + 1) % self._steps


class _BaseMetric(_Base):
    """
    Base class for early termination logic based on the metric measured before.
    """
    def __init__(self, metric: str, for_steps: int):
        """
        Base class for early termination logic based on the metric measured before.
        :param metric: Metric to measure.
        :param for_steps: For how many steps to measure.
        """
        super().__init__(for_steps)
        self._metric = metric

    def _logic(self, break_cb):
        """
        Method that the derived class should overwrite.
        The method is not invoked until sufficient enough measurements are made.
        :param break_cb: Object to call when the algorithm should be interrupted. The population is return implicitly.
        :return: None
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Validate, whether metric and break callback are within keyword parameters and call the termination logic afterward.
        Does not call the logic unless sufficient enough measurements are done.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: `ffeat.STANDARD_REPRESENTATION`
        """
        if self._metric not in kwargs:
            raise ValueError(f"Metric {self._metric} is not in parameters, maybe you forgot to add measure step or the metric is wrong.")
        self._log(kwargs[self._metric])

        if 'ffeat_break' not in kwargs:
            raise ValueError("Key ffeat_break not in dictionary, you need to use it within algorithm class")
        if self._measured >= self._steps:
            self._logic(kwargs['ffeat_break'])

        return super().__call__(*args, **kwargs)
