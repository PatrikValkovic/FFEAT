###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
from typing import Tuple, Any, Dict, Callable
from ffeat import Pipe


class Evaluation(Pipe):
    def __init__(self, evaluation_fn: Callable):
        self.evaluation_fn = evaluation_fn

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        f = self.evaluation_fn(population)
        kwargs.update({"fitness": f})
        return (f, population, *args), kwargs


class EvaluationWrapper:
    Evaluation = Evaluation
