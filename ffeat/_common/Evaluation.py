###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Callable, Tuple, Any, Dict
import torch as t
from ffeat import Pipe


class Evaluation(Pipe):
    def __init__(self, evaluation_fn: Callable):
        self.evaluation_fn = evaluation_fn

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        f = self.evaluation_fn(population)
        kwargs["fitness"] = f
        return (f, population, *args), kwargs


class RowEval(Evaluation):
    def __init__(self, evaluation_fn: Callable):
        super().__init__(
            lambda pop: t.stack([
                evaluation_fn(x) for x in t.unbind(pop, dim=0)
            ], dim=0)
        )

class EvalWrapper:
    Evaluation = Evaluation
    RowEval = RowEval
