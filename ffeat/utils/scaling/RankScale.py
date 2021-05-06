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


class RankScale(Pipe):
    """
    Rewrite the fitness based on the order.
    Individual with lowest value will have `minimum` value and the highest one will equal to `maximum`.
    Remaining values will be distributed linearly in this interval.
    """
    def __init__(self,
                 minimum: Union[_IFU, Callable[..., _IFU]],
                 maximum: Union[_IFU, Callable[..., _IFU]]):
        """
        Rewrite the fitness based on the order.
        Individual with lowest value will have `minimum` value and the highest one will equal to `maximum`.
        Remaining values will be distributed linearly in this interval.
        :param minimum: Minimum value of the new fitness.
        :param maximum: Maximum value of the new fitness.
        """
        self._minimum = self._handle_parameter(minimum)
        self._maximum = self._handle_parameter(maximum)

    def __call__(self, fitnesses, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Rewrite the fitness based on the order.
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

        num = len(fitnesses)
        step = (max - min) / num
        order = t.argsort(fitnesses)
        tmpfitnes = t.arange(0, num, device=fitnesses.device, dtype=fitnesses.dtype)
        tmpfitnes.multiply_(step).add_(min)
        newfitness = t.empty_like(fitnesses)
        newfitness[order] = tmpfitnes
        kwargs[Evaluation.FITNESS_KWORD_ARG] = newfitness

        return (newfitness, *args), kwargs