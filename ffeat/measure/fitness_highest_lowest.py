###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from .FitnessQuantile import FitnessQuantile


class FitnessLowest(FitnessQuantile):
    """
    Measure the lowest fitness value.
    """

    ARG_NAME = "fitness_lowest"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter=None):
        """
        Measure the lowest fitness value.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(0.0, reporter)

    def _dict_key(self):
        """
        Method returning names of all the keyword arguments to insert.
        :return: Iterable of strings.
        """
        yield from super()._dict_key()
        yield FitnessLowest.ARG_NAME


class FitnessHighest(FitnessQuantile):
    """
    Measure the highest fitness value.
    """

    ARG_NAME = "fitness_highest"
    """
    Key under which it is stored in the keyword arguments.
    """

    def __init__(self, reporter=None):
        """
        Measure the highest fitness value.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        super().__init__(1.0, reporter)

    def _dict_key(self):
        """
        Method returning names of all the keyword arguments to insert.
        :return: Iterable of strings.
        """
        yield from super()._dict_key()
        yield FitnessHighest.ARG_NAME
