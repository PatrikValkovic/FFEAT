###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict, Callable
from ffeat import NormalizedPipe


class Lambda(NormalizedPipe):
    """
    Applies lambda function or callable object to all the arguments.
    """
    def __init__(self, _lambda: Callable):
        """
        Applies lambda function or callable object to all the arguments.
        :param _lambda: Lambda function or callable object to apply on arguments.
        """
        self._lambda = _lambda

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Applies lambda function or callable object to arguments.
        :param args: Arguments that are passed down to the lambda function.
        :param kwargs: Keyword arguments passed down to the lambda function.
        :return: Arguments modified by the lambda.
        """
        return super().__call__(self._lambda(*args, **kwargs))
