###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION
from ffeat._common.crossover._Shared import _Shared

_FTU = Union[float, t.Tensor]


class Arithmetic(Pipe, _Shared):
    """
    Arithmetic crossover.
    """
    def __init__(self,
                 offsprings: Union[int, float],
                 parent_weight: Union[_FTU, Callable[..., _FTU]] = None,
                 num_parents: Union[int, Callable[..., int]] = 2,
                 replace_parents: bool = True,
                 in_place: bool = True,
                 discard_parents: bool = False):
        """
        Arithmetic crossover.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param parent_weight: Parents' weights for the crossover. By default 1 / `num_parents`. May be callable and give
        different weights each iteration and for each offspring.
        :param num_parents: Number of parents for one offspring. By default 2.
        :param replace_parents: Whether should offsprings replace their parents. The operator must create the same
        number of offsprings as there are parents in order for this to work. By default true.
        :param in_place: Whether the new population should be created inplace - in the same memory as the original
        population. The new population size must be equal to the old one. By default true.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        _Shared.__init__(self, offsprings, replace_parents, in_place, discard_parents)
        self._num_parents = self._handle_parameter(num_parents)
        self._parent_weight = self._handle_parameter(parent_weight)

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply arithmetic crossover on the population.
        :param population: Tensor with population. Expect first dimension to enumerate over individuals.
        :param args: Arguments to pass along.
        :param kwargs: Keyword arguments to pass along.
        :return: Population with offsprings integrate.
        """
        itp = t.long
        ptp = population.dtype
        dev = population.device
        pop_len = len(population)
        dim = population.shape[1:]
        num_children = self._offsprings
        num_children = num_children if isinstance(num_children, int) else int(pop_len * num_children)
        assert isinstance(num_children, int), f"Number of offsprings should be int, {type(num_children)} received"
        num_parents = self._num_parents(population, *args, **kwargs)
        num_parents = int(num_parents) if int(num_parents) == num_parents else num_parents
        assert isinstance(num_parents, int), f"Number of parents should be int, {type(num_parents)} received"

        parent_weights = self._parent_weight(population, *args, **kwargs)
        if parent_weights is None:
            parent_weights = 1.0 / num_parents
        if isinstance(parent_weights, float):
            parent_weights = t.tensor(parent_weights, dtype=ptp, device=dev)
        assert isinstance(parent_weights, t.Tensor), f"Parent weights should be tensor, {type(parent_weights)} received"
        # num_offsprings, num_parents, dimension
        NoneDim = [1] * len(dim)
        if len(parent_weights.shape) == 0:
            parent_weights = parent_weights.reshape(tuple([1,1, *NoneDim]))
        elif len(parent_weights.shape) == 1 and parent_weights.shape[0] == num_parents:
            parent_weights = parent_weights.reshape(tuple([
                1,
                parent_weights.shape[0],
                *NoneDim
            ]))
        elif len(parent_weights.shape) == 1 and parent_weights.shape[0] == num_children:
            parent_weights = parent_weights.reshape(tuple([
                parent_weights.shape[0],
                1,
                *NoneDim
            ]))
        elif len(parent_weights.shape) == 2 and parent_weights.shape == (num_children, num_parents):
            parent_weights = parent_weights.reshape(tuple([
                num_children, num_parents, *NoneDim
            ]))
        elif len(parent_weights.shape) == len(dim) and parent_weights.shape == dim:
            parent_weights = parent_weights.reshape(tuple([
                1,1,
                *dim
            ]))
        elif len(parent_weights.shape) == 1+len(dim) and parent_weights.shape[0] == num_parents and parent_weights.shape[1:] == dim:
            parent_weights = parent_weights.reshape(tuple([
                1,
                num_parents,
                *dim
            ]))
        elif len(parent_weights.shape) == 1+len(dim) and parent_weights.shape[0] == num_children and parent_weights.shape[1:] ==  dim:
            parent_weights = parent_weights.reshape(tuple([
                num_children,
                1,
                *dim
            ]))

        assert len(parent_weights.shape) == 2+len(dim), f"Expected weights in {2+len(dim)} dimension, {len(parent_weights.shape)} given"
        assert parent_weights.shape[0] in {1,num_children}, f"Expected first dimension of weights to be {num_children}, {parent_weights.shape[0]} given"
        assert parent_weights.shape[1] in {1,num_parents}, f"Expected second dimension of weights to be {num_parents}, {parent_weights.shape[1]} given"
        for _di, _d in enumerate(dim):
            assert parent_weights.shape[2+_di] in {1,_d}, f"Expected dimension {2+_di} of weights to be {_d}, {parent_weights.shape[2+_di]} given"

        parents_indices = t.randint(pop_len, (num_children,num_parents), dtype=itp, device=dev)
        parents = population[parents_indices]
        parents.multiply_(parent_weights)
        children = t.sum(parents, dim=1)

        pop = self._handle_pop(population, children, parents_indices[:,0])

        return (pop, *args), kwargs
