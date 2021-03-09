###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Callable
from . import Lambda

class EachArg(Lambda):
    def __init__(self, _lambda: Callable):
        super().__init__(_lambda)

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            narg, _ = super().__call__(args[i], **kwargs)
            args[i] = narg[0]
        return tuple(args), kwargs
