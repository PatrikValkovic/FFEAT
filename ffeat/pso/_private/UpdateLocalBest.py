###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION


class UpdateLocalBest(Pipe):
    """
    Update particles' local best positions.
    """

    def __call__(self, fitnesses, position, fitness_lbest, positions_lbest, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Updated global best positions. Expects all the arguments to use first dimension to enumerate over particles and
        the parameters are in the same order.
        :param fitnesses: Fitness of the particles.
        :param position: Position of the particles.
        :param fitness_lbest: Current fitness of global best positions.
        :param positions_lbest: Current global best positions.
        :param kwargs: Keyword arguments.
        :return: Updated fitness of local best positions, at the positions themselves.
        """
        better = fitnesses < fitness_lbest
        fitness_lbest = t.minimum(fitnesses, fitness_lbest, out=fitness_lbest)
        positions_lbest[better] = position[better]
        return (fitness_lbest, positions_lbest), {}
