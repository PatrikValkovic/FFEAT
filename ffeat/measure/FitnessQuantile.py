###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from typing import Union
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from .Base import Base


class FitnessQuantile(Pipe, Base):
    """
    Measure the fitness quantile.
    """

    def __init__(self, quantile: Union[float, t.Tensor], reporter = None):
        """
        Measure the fitness quantile.
        :param quantile: Quantile to measure.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        Base.__init__(self, reporter)
        self.quenatile = quantile

    def _dict_key(self):
        """
        Method returning names of all the keyword arguments to insert.
        :return: Iterable of strings.
        """
        percentage = self.quenatile * 100
        if int(percentage) == percentage:
            yield f"fitness_q{int(percentage):02}"
        yield f"fitness_q{self.quenatile}"

    def __call__(self, fitnesses, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Measure the fitness quantile and add it as the keyword arguments.
        :param fitnesses: Fitness expected to be one dimension array.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Arguments and passed arguments with the fitness.
        """
        q = float(t.quantile(fitnesses, self.quenatile))
        self._report(q)
        for k in self._dict_key():
            kwargs.update({k: q})
        return (fitnesses, *args), kwargs


class FitnessMedian(FitnessQuantile):
    """
    Measure the fitness median.
    """

    ARG_NAME = "fitness_median"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter = None):
        """
        Measure the fitness median.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.5, reporter)

    def _dict_key(self):
        """
        Method returning names of all the keyword arguments to insert.
        :return: Iterable of strings.
        """
        yield from super()._dict_key()
        yield FitnessMedian.ARG_NAME


class Fitness95Quantile(FitnessQuantile):
    """
    Measure the fitness 0.95 quantile.
    """

    ARG_NAME = "fitness_q95"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter = None):
        """
        Measure the fitness 0.95 quantile.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.95, reporter)


class Fitness99Quantile(FitnessQuantile):
    """
    Measure the fitness 0.99 quantile.
    """

    ARG_NAME = "fitness_q99"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter = None):
        """
        Measure the fitness 0.99 quantile.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.99, reporter)


class Fitness05Quantile(FitnessQuantile):
    """
    Measure the fitness 0.05 quantile.
    """

    ARG_NAME = "fitness_q05"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter = None):
        """
        Measure the fitness 0.05 quantile.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.05, reporter)


class Fitness01Quantile(FitnessQuantile):
    """
    Measure the fitness 0.01 quantile.
    """

    ARG_NAME = "fitness_q01"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter = None):
        """
        Measure the fitness 0.01 quantile.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.01, reporter)
