###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ffeat._common.Evaluation import Evaluation


class MultiplicativeInverse(Pipe):
    """
    Scale the fitness by the formula new_x = 1 / x. Transform minimisation problem into maximisation one and vice versa.
    """
    def __call__(self, fitnesses, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Scale the fitness by the formula new_x = 1 / x. Transform minimisation problem into maximisation one and vice versa.
        :param fitnesses: Fitness values of the population.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: `ffeat.STANDARD_REPRESENTATION` where first argument is scaled fitness and rest is passed along.
        """
        fitnesses = t.divide(t.tensor(1.0), fitnesses, out=fitnesses if fitnesses.dtype.is_floating_point else None)
        kwargs[Evaluation.FITNESS_KWORD_ARG] = fitnesses
        return (fitnesses, *args), kwargs