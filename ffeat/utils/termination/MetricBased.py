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
        if self._measured == self._steps and np.argmin(self._values) == self._current_index:
            break_cb()


class StdBellow(_BaseMetric):
    def __init__(self, metric: str, step_into_account: int, threshold_value: float):
        super().__init__(metric, step_into_account)
        self._threshold = threshold_value

    def _logic(self, break_cb):
        if self._measured == self._steps and np.std(self._vals) < self._threshold:
            break_cb()


class MetricReached(_BaseMetric):
    def __init__(self, metric: str, for_steps: int, target_value: float, minimizations = True):
        super().__init__(metric, for_steps)
        self._target = target_value
        self._minimization = minimizations
        self._satisfied_times = 0

    def _logic(self, break_cb):
        val = self._values[(self._current_index + self._steps - 1) % self._steps]
        if val < self._target and self._minimization or val > self._target and not self._minimization:
            self._satisfied_times += 1
            if self._satisfied_times > self._steps:
                break_cb()
        else:
            self._satisfied_times = 0
