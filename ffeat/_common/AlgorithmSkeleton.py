###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Tuple, Any, Dict, Optional
from ffeat import Pipe, flow


class AlgorithmSkeleton(Pipe):
    def __init__(self,
                 initialization: Pipe,
                 *steps: Pipe,
                 iterations: Optional[int] = 100):
        self.__flow = flow.Sequence(
            initialization,
            flow.Repeat(
                flow.Sequence(
                    *steps
                ),
                max_iterations=iterations,
                loop_arguments=True,
                identifier='ffeat'
            )
        )

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return self.__flow(*args, **kwargs)
