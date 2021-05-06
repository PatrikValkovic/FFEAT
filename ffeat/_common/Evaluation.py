###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION


class Evaluation(Pipe):
    """
    Evaluate the whole population at once.
    """
    FITNESS_KWORD_ARG = "orig_fitness"
    """
    Key under which it the fitness stored in keyword arguments.
    """

    def __init__(self, evaluation_fn: Callable):
        """
        Evaluate the whole population at once.
        :param evaluation_fn: Evaluation function. Should receive all the arguments and return a single array of values
        representing fitness.
        """
        self.evaluation_fn = evaluation_fn

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Evaluates the population and return their fitness as the first arguments.
        Also put the fitness as `orig_fitness` keyword arguments.
        :param population: Population to evaluate.
        :param args: Additional arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Population fitness, Population, and rest of the arguments.
        """
        f = self.evaluation_fn(population)
        kwargs[Evaluation.FITNESS_KWORD_ARG] = f
        return (f, population, *args), kwargs


class RowEval(Evaluation):
    """
    Evaluate the population by calling the evaluation function on each individual.
    """
    def __init__(self, evaluation_fn: Callable):
        """
        Evaluate the population by calling the evaluation function on each individual.
        :param evaluation_fn: Function to be called on each individual.
        """
        super().__init__(
            lambda pop: t.stack([
                evaluation_fn(x) for x in t.unbind(pop, dim=0)
            ], dim=0)
        )


class EvalWrapper:
    """
    Class that summary the evaluation implementations.
    """
    Evaluation = Evaluation
    RowEval = RowEval
