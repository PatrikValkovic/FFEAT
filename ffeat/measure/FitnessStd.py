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


class FitnessStd(Pipe, Base):
    def __init__(self, reporter=None):
        Base.__init__(self, reporter)

    def __call__(self, fitness, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        m = float(t.std(fitness))
        self._report(m)
        kwargs.update({"fitness_std": m})
        return (fitness, *args), kwargs
