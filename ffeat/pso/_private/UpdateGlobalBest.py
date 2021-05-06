###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ..neighborhood.Neighborhood import Neighborhood


class UpdateGlobalBest(Pipe):
    """
    Update particles' global best positions.
    """

    def __init__(self, neighborhood: Neighborhood):
        """
        Update particles' global best positions.
        :param neighborhood: Particles neighborhood definition.
        """
        self._neighborhood = neighborhood

    def __call__(self, fitnesses, position, fitness_gbest, positions_gbest, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Updated global best positions. Expects all the arguments to use first dimension to enumerate over particles and
        the parameters are in the same order.
        :param fitnesses: Fitness of the particles.
        :param position: Position of the particles.
        :param fitness_gbest: Current fitness of global best positions.
        :param positions_gbest: Current global best positions.
        :param kwargs: Keyword arguments.
        :return: Updated fitness of global best positions, at the positions themselves.
        """
        neighborhood_indices = self._neighborhood(fitnesses, position, **kwargs)
        neighborhood_fitness = fitnesses[neighborhood_indices]
        fitness_best_neighborhood_indices = t.argmin(neighborhood_fitness, dim=1)
        best_indices = neighborhood_indices[range(len(neighborhood_indices)), fitness_best_neighborhood_indices]
        better_global = fitnesses[best_indices] < fitness_gbest
        fitness_gbest = t.minimum(fitness_gbest, fitnesses[best_indices], out=fitness_gbest)
        positions_gbest[better_global] = position[best_indices[better_global]]
        return (fitness_gbest, positions_gbest), {}
