###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
from typing import Optional
from ffeat import Pipe, flow, STANDARD_REPRESENTATION


class AlgorithmSkeleton(Pipe):
    """
    Basic algorithm skeleton for Genetic Algorithms and Real-Coded Evolutionary Algorithms.
    """
    def __init__(self,
                 initialization: Pipe,
                 *steps: Pipe,
                 iterations: Optional[int] = 100):
        """
        Basic algorithm skeleton for Genetic Algorithms and Real-Coded Evolutionary Algorithms.
        :param initialization: Pipe that initialize the population.
        :param steps: Crossover operators applied in loop.
        :param iterations: Number of iterations. For None value the loop must be early terminated.
        """
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

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Starts the algorithm.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: Output of the last iteration. Usually just consists of population and keyword arguments.
        """
        return self.__flow(*args, **kwargs)
