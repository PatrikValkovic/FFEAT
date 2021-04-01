###############################
#
# Created by Patrik Valkovic
# 01.04.21
#
###############################
import torch as t
from typing import Union


class _Shared:
    def __init__(self,
                 offsprings: Union[int, float],
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        self._offsprings = offsprings
        self.replace_parents = replace_parents
        self.in_place = in_place
        self.discard_parents = discard_parents

    def _handle_pop(self, population, children, parents_indices):
        if self.discard_parents:
            return children
        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices.flatten()] = children
        else:
            population = t.cat([population, children], dim=0)
        return population
