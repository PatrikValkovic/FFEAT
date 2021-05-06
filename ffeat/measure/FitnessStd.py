###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from .Base import Base


class FitnessStd(Pipe, Base):
    """
    Measure the standard deviation of the fitness.
    """

    ARG_NAME = "fitness_std"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter=None):
        """
        Measure the standard deviation of the fitness.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        Base.__init__(self, reporter)

    def __call__(self, fitnesses, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Measure the standard deviation of the fitness add it as the keyword arguments.
        :param fitnesses: Fitness expected to be one dimension array.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Arguments and passed arguments with the fitness.
        """
        m = float(t.std(fitnesses))
        self._report(m)
        kwargs[FitnessStd.ARG_NAME] = m
        return (fitnesses, *args), kwargs
