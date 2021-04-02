###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from .FitnessQuantile import FitnessQuantile


class FitnessLowest(FitnessQuantile):
    ARG_NAME = "fitness_lowest"

    def __init__(self, reporter=None):
        super().__init__(0.0, reporter)

    def _dict_key(self):
        yield from super()._dict_key()
        yield FitnessLowest.ARG_NAME


class FitnessHighest(FitnessQuantile):
    ARG_NAME = "fitness_highest"

    def __init__(self, reporter=None):
        super().__init__(1.0, reporter)

    def _dict_key(self):
        yield from super()._dict_key()
        yield FitnessHighest.ARG_NAME
