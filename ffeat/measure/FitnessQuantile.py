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
    def __init__(self, quantile: Union[float, t.Tensor], reporter = None):
        Base.__init__(self, reporter)
        self.quenatile = quantile

    def _dict_key(self):
        percentage = self.quenatile * 100
        if int(percentage) == percentage:
            yield f"fitness_q{int(percentage):02}"
        yield f"fitness_q{self.quenatile}"

    def __call__(self, fitnesses, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        q = float(t.quantile(fitnesses, self.quenatile))
        self._report(q)
        for k in self._dict_key():
            kwargs.update({k: q})
        return (fitnesses, *args), kwargs


class FitnessMedian(FitnessQuantile):
    ARG_NAME = "fitness_median"

    def __init__(self, reporter = None):
        super().__init__(0.5, reporter)

    def _dict_key(self):
        yield from super()._dict_key()
        yield FitnessMedian.ARG_NAME


class Fitness95Quantile(FitnessQuantile):
    ARG_NAME = "fitness_q95"

    def __init__(self, reporter = None):
        super().__init__(0.95, reporter)


class Fitness99Quantile(FitnessQuantile):
    ARG_NAME = "fitness_q99"

    def __init__(self, reporter = None):
        super().__init__(0.99, reporter)


class Fitness05Quantile(FitnessQuantile):
    ARG_NAME = "fitness_q05"

    def __init__(self, reporter = None):
        super().__init__(0.05, reporter)


class Fitness01Quantile(FitnessQuantile):
    ARG_NAME = "fitness_q01"

    def __init__(self, reporter = None):
        super().__init__(0.01, reporter)
