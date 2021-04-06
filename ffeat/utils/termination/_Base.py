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
    def __init__(self,
                 for_steps: int):
        self._steps = for_steps
        self._current_index = 0
        self._measured = 0
        self._values = np.zeros(self._steps, dtype=float)

    def _log(self, value):
        self._values[self._current_index] = float(value)
        self._measured = min(self._measured + 1, self._steps)
        self._current_index = (self._current_index + 1) % self._steps

class _BaseMetric(_Base):
    def __init__(self, metric: str, for_steps: int):
        super().__init__(for_steps)
        self._metric = metric

    def _logic(self, break_cb):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if self._metric not in kwargs:
            raise ValueError(f"Metric {self._metric} is not in parameters, maybe you forgot to add measure step or the metric is wrong.")
        self._log(kwargs[self._metric])

        if 'ffeat_break' not in kwargs:
            raise ValueError("Key ffeat_break not in dictionary, you need to use it within algorithm class")
        if self._measured >= self._steps:
            self._logic(kwargs['ffeat_break'])

        return super().__call__(*args, **kwargs)
