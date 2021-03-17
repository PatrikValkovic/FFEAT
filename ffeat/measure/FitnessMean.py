###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe
from .Base import Base


class FitnessMean(Pipe, Base):
    ARG_NAME = "fitness_mean"

    def __init__(self, reporter = None):
        Base.__init__(self, reporter)

    def __call__(self, fitnesses, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        m = float(t.mean(fitnesses))
        self._report(m)
        kwargs[FitnessMean.ARG_NAME] = m
        return (fitnesses, *args), kwargs
