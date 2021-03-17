###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
import numpy as np
from ._Base import _BaseMetric


class NoImprovement(_BaseMetric):
    def _logic(self, break_cb):
        if self._measured == self._steps and np.argmin(self._vals) == self._current_index:
            break_cb()


class StdBellow(_BaseMetric):
    def __init__(self, metric: str, step_into_account: int, threshold_value: float):
        super().__init__(metric, step_into_account)
        self._threshold = threshold_value

    def _logic(self, break_cb):
        if self._measured == self._steps and np.std(self._vals) < self._threshold:
            break_cb()


class MetricReached(_BaseMetric):
    def __init__(self, metric: str, for_steps: int, target_value: float):
        super().__init__(metric, for_steps)
        self._target = target_value

    def _logic(self, break_cb):
        if self._vals[(self._current_index + self._steps - 1) % self._steps] <= self._target:
            break_cb()
