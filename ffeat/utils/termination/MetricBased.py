###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
import numpy as np
from ._Base import _BaseMetric


class NoImprovement(_BaseMetric):
    """
    Interrupt the algorithm if the metric does not improve for a specified number of steps.
    """
    def _logic(self, break_cb):
        """
        Interrupt the algorithm if the metric does not improve for a specified number of steps.
        :param break_cb: Object to call when the algorithm should be interrupted. The population is return implicitly.
        :return: None
        """
        # current index is the one the implementation should write in the next iteration, therefore the oldest measured value.
        if self._measured == self._steps and np.argmin(self._values) == self._current_index:
            break_cb()


class StdBellow(_BaseMetric):
    """
    Terminate the algorithm if metric standard deviation is bellow given threshold for a given number of steps.
    """
    def __init__(self, metric: str, step_into_account: int, threshold_value: float):
        """
        Terminate the algorithm if metric standard deviation is bellow given threshold for a given number of steps.
        :param metric: Metric to measure.
        :param step_into_account: How many steps in the history the algorithm should look.
        :param threshold_value: The threshold value bellow which the algorithm will terminate.
        """
        super().__init__(metric, step_into_account)
        self._threshold = threshold_value

    def _logic(self, break_cb):
        """
        Terminate the algorithm if metric standard deviation is bellow given threshold for a given number of steps.
        :param break_cb: Object to call when the algorithm should be interrupted. The population is return implicitly.
        :return: None
        """
        if self._measured == self._steps and np.std(self._values) < self._threshold:
            break_cb()


class MetricReached(_BaseMetric):
    """
    Terminate the algorithm if given metric reach given threshold and keep it for given number of steps.
    """
    def __init__(self, metric: str, for_steps: int, target_value: float, minimization = True):
        """
        Terminate the algorithm if given metric reach given threshold and keep it for given number of steps.
        :param metric: Metric to measure.
        :param for_steps: How many steps should be the metric bellow or above threshold.
        If the algorithm should end immediately, pass value 1.
        :param target_value: Threshold value to reach.
        :param minimization: Whether is it minimisation (default) or maximization problem.
        """
        super().__init__(metric, for_steps)
        self._target = target_value
        self._minimization = minimization

    def _logic(self, break_cb):
        """
        Terminate the algorithm if given metric reach given threshold and keep it for given number of steps.
        :param break_cb: Object to call when the algorithm should be interrupted. The population is return implicitly.
        :return: None
        """
        if self._minimization:
            satisfied = self._values < self._target
        else:
            satisfied = self._values > self._target
        if np.all(satisfied):
                break_cb()
