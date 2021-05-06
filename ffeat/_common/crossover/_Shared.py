###############################
#
# Created by Patrik Valkovic
# 01.04.21
#
###############################
import torch as t
from typing import Union


class _Shared:
    """
    Shared logic for crossover operators. Implements logic regarding crossover schemes.
    To use comma schema, set `discard_parents` to true.
    To use plus schema, set `replace_parents` and `discard_parents` to false.
    To use default schema (most efficient), use `discard_parents` to false and `replace_parents` to true.
    """
    def __init__(self,
                 offsprings: Union[int, float],
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        """
        Shared logic for crossover operators. Implements logic regarding crossover schemes.
        To use comma schema, set `discard_parents` to true.
        To use plus schema, set `replace_parents` and `discard_parents` to false.
        To use default schema (most efficient), use `discard_parents` to false and `replace_parents` to true.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param replace_parents: Whether should offsprings replace their parents. The operator must create the same
        number of offsprings as there are parents in order for this to work. By default true.
        :param in_place: Whether the new population should be created inplace - in the same memory as the original
        population. The new population size must be equal to the old one. By default true.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        self._offsprings = offsprings
        self.replace_parents = replace_parents
        self.in_place = in_place
        self.discard_parents = discard_parents

    def _handle_pop(self, population, children, parents_indices):
        """
        Method that handles the offspring placement.
        :param population: Tensor with the old population. Expect first dimension to enumerate over individuals.
        :param children: Tensor with the offsprings. Expect first dimension to enumerate over individuals.
        :param parents_indices: Indicies of parents. Expect one dimensional array with one parent for each offspring.
        Offsprings will replace parents based on this indices.
        :return: Population with offsprings integrated.
        """
        if self.discard_parents:
            return children
        population = t.clone(population) if not self.in_place and self.replace_parents else population
        if self.replace_parents:
            population[parents_indices.flatten()] = children
        else:
            population = t.cat([population, children], dim=0)
        return population
