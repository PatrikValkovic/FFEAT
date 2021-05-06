###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Callable
from ffeat import STANDARD_REPRESENTATION
from . import Lambda

class EachArg(Lambda):
    """
    Applies lambda function to each parameter.
    """
    def __init__(self, _lambda: Callable):
        """
        Applies lambda function to each parameter.
        :param _lambda: Lambda function to callable object to apply to each parameter.
        """
        super().__init__(_lambda)

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply callable object to each parameter.
        :param args: Arguments on which to apply the operation.
        :param kwargs: Keyword arguments. Not changed.
        :return: Arguments after lambda applied with keyword arguments unchanged.
        """
        args = list(args)
        for i in range(len(args)):
            narg, _ = super().__call__(args[i], **kwargs)
            args[i] = narg[0]
        return tuple(args), kwargs
