###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ffeat._common.Evaluation import Evaluation

_IFU = Union[int, float]


class LinearScale(Pipe):
    """
    Scale the fitness linearly, such that the smallest value is equal `minimum`, and the highest `maximum`.
    """
    def __init__(self,
                 minimum: Union[_IFU, Callable[..., _IFU]],
                 maximum: Union[_IFU, Callable[..., _IFU]]):
        """
        Scale the fitness linearly, such that the smallest value is equal `minimum`, and the highest `maximum`.
        :param minimum: Minimum value of the new fitness.
        :param maximum: Maximum value of the new fitness.
        """
        self._minimum = self._handle_parameter(minimum)
        self._maximum = self._handle_parameter(maximum)

    def __call__(self, fitnesses, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Scale the fitness linearly.
        :param fitnesses: Fitness values of the population.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: `ffeat.STANDARD_REPRESENTATION` where first argument is scaled fitness and rest is passed along.
        """
        min = self._minimum(fitnesses, *args, **kwargs)
        if not isinstance(min, float):
            min = float(min)
        max = self._maximum(fitnesses, *args, **kwargs)
        if not isinstance(max, float):
            max = float(max)

        fmin, fmax = t.min(fitnesses), t.max(fitnesses)
        fitnesses = t.sub(fitnesses, fmin, out=fitnesses)
        fitnesses = t.multiply(fitnesses, (max - min) / fmax, out=fitnesses)
        fitnesses = t.add(fitnesses, min, out=fitnesses)
        kwargs[Evaluation.FITNESS_KWORD_ARG] = fitnesses

        return (fitnesses, *args), kwargs


