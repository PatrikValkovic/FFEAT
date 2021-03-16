###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable
import torch as t
from ffeat import Pipe

_FDU = Union[float, t.distributions.Distribution]


class Differential(Pipe):
    def __init__(self,
                 num_offsprings: int = None,
                 fraction_offsprings: float = None,
                 crossover_probability: Union[_FDU, Callable[..., _FDU]] = 0.9,
                 differential_weight: Union[_FDU, Callable[..., _FDU]] = 0.8,
                 evaluation=None,
                 replace_parents: bool = True,
                 replace_only_better: bool = False):
        if num_offsprings is None and fraction_offsprings is None:
            raise ValueError("Either number of offsprings or a percentage must be provided")
        if replace_only_better and evaluation is None:
            raise ValueError("If you want to replace with better, evaluation must be provided")
        self._num_offsprings = num_offsprings
        self._fraction_offsprings = fraction_offsprings
        self._CR = self._handle_parameter(crossover_probability)
        self._F = self._handle_parameter(differential_weight)
        self._replace_parents = replace_parents
        self._replace_only_better = replace_only_better
        self._evaluate = evaluation

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        itp = t.long
        ptp = population.dtype
        dev = population.device
        pop_len = len(population)
        dim = population.shape[1:]
        num_children = self._num_offsprings if self._num_offsprings is not None else int(pop_len * self._fraction_offsprings)
        CR = self._CR(population, *args, **kwargs)
        if not isinstance(CR, t.distributions.Distribution):
            CR = t.distributions.Uniform(CR-1e-6, CR+1e-6)
        F = self._F(population, *args, **kwargs)
        if not isinstance(F, t.distributions.Distribution):
            F = t.distributions.Uniform(F-1e-6, F+1e-6)

        parents_probs = t.ones((num_children, pop_len), dtype=t.float32, device=dev)
        parents_probs = t.divide(parents_probs, t.tensor(pop_len), out=parents_probs)
        parent_indices = t.multinomial(parents_probs, 4, replacement=False).T  # Unif[P0,P1 + F(P2 - P3); CR]
        CR_sample = CR.sample((num_children,)).to(dev).type(t.float32)
        F_sample = F.sample((num_children,)).to(dev).type(t.float32)

        children = t.clone(population[parent_indices[2]])
        children = t.subtract(children, population[parent_indices[3]], out=children)
        children = t.multiply(children, F_sample.reshape(-1, *[1] * len(dim)), out=children)
        children = t.add(children, population[parent_indices[1]], out=children)
        mut = t.rand((num_children, *dim), device=dev) > CR_sample.reshape(-1, *[1] * len(dim))
        children[mut] = population[parent_indices[0]][mut]

        if self._replace_only_better:
            (parent_f, *_), _ = self._evaluate(population[parent_indices[0]])
            (children_f, *_), _ = self._evaluate(children)
            children_better = children_f < parent_f
            population[parent_indices[0][children_better]] = children[children_better]
        elif not self._replace_parents:
            population = t.cat([
                population,
                children
            ], dim=0)
        else:
            population[parent_indices[0]] = children

        return (population, *args), kwargs
