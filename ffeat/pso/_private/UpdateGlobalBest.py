###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Any, Dict
import torch as t
from ffeat import Pipe
from ..neighborhood.Neighborhood import Neighborhood


class UpdateGlobalBest(Pipe):
    def __init__(self, neighborhood: Neighborhood):
        self._neighborhood = neighborhood

    def __call__(self, fitnesses, position, fitness_gbest, positions_gbest, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        neighborhood_indices = self._neighborhood(fitnesses, position, **kwargs)
        neighborhood_fitness = fitnesses[neighborhood_indices]
        fitness_best_neighborhood_indices = t.argmin(neighborhood_fitness, dim=1)
        best_indices = neighborhood_indices[range(len(neighborhood_indices)), fitness_best_neighborhood_indices]
        better_global = fitnesses[best_indices] < fitness_gbest
        fitness_gbest = t.minimum(fitness_gbest, fitnesses[best_indices], out=fitness_gbest)
        positions_gbest[better_global] = position[best_indices[better_global]]
        return (fitness_gbest, positions_gbest), {}
