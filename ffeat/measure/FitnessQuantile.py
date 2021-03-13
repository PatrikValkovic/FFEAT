###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from typing import Tuple, Any, Dict, Union
import torch as t
from ffeat import Pipe
from .Base import Base


class FitnessQuantile(Pipe, Base):
    def __init__(self, quantile: Union[float, t.Tensor], reporter):
        super(Base, self).__init__(reporter)
        self.quenatile = quantile

    def __call__(self, fitness, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        self._report(t.quantile(fitness, self.quenatile))
        return (fitness, *args), kwargs


class FitnessMedian(FitnessQuantile):
    def __init__(self, reporter):
        super().__init__(0.5, reporter)


class Fitness95Quantile(FitnessQuantile):
    def __init__(self, reporter):
        super().__init__(0.95, reporter)


class Fitness99Quantile(FitnessQuantile):
    def __init__(self, reporter):
        super().__init__(0.99, reporter)
