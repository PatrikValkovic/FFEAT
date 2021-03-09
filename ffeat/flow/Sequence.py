###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from ffeat import Pipe


class Sequence(Pipe):
    def __init__(self, *steps: Pipe):
        self.__steps = steps

    def __call__(self, *args, **kwargs):
        for step in self.__steps:
            args, kwargs = step(*args, **kwargs)
        return args, kwargs
