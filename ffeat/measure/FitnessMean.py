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
    def __init__(self, reporter):
        super(Base, self).__init__(reporter)

    def __call__(self, fitness, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        self._report(t.mean(fitness))
        return (fitness, *args), kwargs
