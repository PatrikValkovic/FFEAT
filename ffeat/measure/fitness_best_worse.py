###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
from .FitnessQuantile import FitnessQuantile


class FitnessBest(FitnessQuantile):
    def __init__(self, reporter=None):
        super().__init__(1.0, reporter)

    def _dict_key(self):
        yield from super()._dict_key()
        yield "fitness_best"


class FitnessWorse(FitnessQuantile):
    def __init__(self, reporter=None):
        super().__init__(0.0, reporter)

    def _dict_key(self):
        yield from super()._dict_key()
        yield "fitness_worse"
