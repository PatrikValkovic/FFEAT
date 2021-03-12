###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe, flow


# TODO expand
class Strategy(Pipe):
    def __init__(self,
                 initialization: Pipe,
                 *steps: Pipe):
        self.__flow = flow.Sequence(
            initialization,
            flow.Repeat(
                flow.Sequence(
                    *steps
                ),
                max_iterations=100,
                loop_arguments=True
            )
        )

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return self.__flow(*args, **kwargs)