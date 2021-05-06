###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION

_FDU = Union[float, t.distributions.Distribution]


class _Differential(Pipe):
    """
    Differential evolution operator. May be used as the whole evolution or as a single crossover operator.
    """
    def __init__(self,
                 parent_fitnesses,
                 report_offspring_fitness,
                 offsprings: Union[int, float],
                 crossover_probability: Union[_FDU, Callable[..., _FDU]] = 0.9,
                 differential_weight: Union[_FDU, Callable[..., _FDU]] = 0.8,
                 evaluation=None,
                 replace_parents: bool = True,
                 replace_only_better: bool = False,
                 discard_parents: bool = False):
        """
        Differential evolution operator. May be used as the whole evolution or as a single crossover operator.
        :param parent_fitnesses: Callable to get parent fitness.
        :param report_offspring_fitness: Callable to report fitness of the offsprings.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param crossover_probability: Probability that offspring inherit gene from donor vector.
        :param differential_weight: Direction strength parameter.
        :param evaluation: Evaluation pipe to evaluate new offsprings. Needed only if `replace_only_better` is true.
        :param replace_parents: Whether should offsprings replace their children (default) or concatenate themselves to the population.
        :param replace_only_better: Whether to replace parent only if offsprings is better. Default false.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        if replace_only_better and evaluation is None:
            raise ValueError("If you want to replace with better, evaluation must be provided")
        self._offsprings = offsprings
        self._CR = self._handle_parameter(crossover_probability)
        self._F = self._handle_parameter(differential_weight)
        self._replace_parents = replace_parents
        self._replace_only_better = replace_only_better
        self._discard_parents = discard_parents
        self._evaluate = evaluation
        self.__parent_fitnesses = parent_fitnesses
        self.__report_offspring_fitness = report_offspring_fitness

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply the differential operator on the population and return it.
        :param population: Tensor representing the population.
        Expects the first dimension enumerate over the individuals.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Return fitness and mutated population.
        """
        dev = population.device
        pop_len = len(population)
        dim = population.shape[1:]
        num_children = self._offsprings if isinstance(self._offsprings, int) else int(pop_len * self._offsprings)
        CR = self._CR(population, *args, **kwargs)
        if not isinstance(CR, t.distributions.Distribution):
            CR = t.distributions.Uniform(CR-1e-6, CR+1e-6)
        F = self._F(population, *args, **kwargs)
        if not isinstance(F, t.distributions.Distribution):
            F = t.distributions.Uniform(F-1e-6, F+1e-6)

        parents_probs = t.tensor(1 / pop_len, dtype=t.float32, device=dev).as_strided_((num_children, pop_len), (0,0))
        parent_indices = t.multinomial(parents_probs, 4, replacement=False).T  # Unif[P0,P1 + F(P2 - P3); CR]
        CR_sample = CR.sample((num_children,)).to(dev).type(t.float32)
        F_sample = F.sample((num_children,)).to(dev).type(t.float32)

        children = t.clone(population[parent_indices[2]])
        children = t.subtract(children, population[parent_indices[3]], out=children)
        children = t.multiply(children, F_sample.reshape(-1, *[1] * len(dim)), out=children)
        children = t.add(children, population[parent_indices[1]], out=children)
        mut = t.rand((num_children, *dim), device=dev) > CR_sample.reshape(-1, *[1] * len(dim))
        children[mut] = population[parent_indices[0]][mut]

        if self._discard_parents:
            population = children
        elif self._replace_only_better:
            (parent_f, *_), _ = self.__parent_fitnesses(population, parent_indices[0], *args, **kwargs)
            (children_f, *_), _ = self._evaluate(children)
            children_better = children_f < parent_f
            parents_to_replace = parent_indices[0][children_better]
            population[parents_to_replace] = children[children_better]
            self.__report_offspring_fitness(parents_to_replace, children_f, children_better)
        elif not self._replace_parents:
            population = t.cat([
                population,
                children
            ], dim=0)
            self.__report_offspring_fitness()
        else:
            population[parent_indices[0]] = children
            self.__report_offspring_fitness(parent_indices[0])

        return (population, *args), kwargs


class Differential(_Differential):
    """
    Differential evolution operator.  May be used as the whole evolution or as a single crossover operator.
    In case you have parents fitness available, use the `DifferentialWithFitness` class.
    """
    def __init__(self,
                 offsprings: Union[int, float],
                 crossover_probability: Union[_FDU, Callable[..., _FDU]] = 0.9,
                 differential_weight: Union[_FDU, Callable[..., _FDU]] = 0.8,
                 evaluation=None,
                 replace_parents: bool = True,
                 replace_only_better: bool = False,
                 discard_parents: bool = False):
        """
        Differential evolution operator.  May be used as the whole evolution or as a single crossover operator.
        In case you have parents fitness available, use the `DifferentialWithFitness` class.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param crossover_probability: Probability that offspring inherit gene from donor vector.
        :param differential_weight: Direction strength parameter.
        :param evaluation: Evaluation pipe to evaluate new offsprings. Needed only if `replace_only_better` is true.
        :param replace_parents: Whether should offsprings replace their children (default) or concatenate themselves to the population.
        :param replace_only_better: Whether to replace parent only if offsprings is better. Default false.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        super().__init__(lambda parents, indices, *args, **kwargs: evaluation(parents[indices], *args, **kwargs),
                         lambda *args, **kwargs: None,
                         offsprings, crossover_probability,
                         differential_weight, evaluation,
                         replace_parents, replace_only_better, discard_parents)


class DifferentialWithFitness(_Differential):
    """
    Differential evolution operator.  May be used as the whole evolution or as a single crossover operator.
    The parents fitness must be available.
    """
    def __init__(self,
                 offsprings: Union[int, float],
                 crossover_probability: Union[_FDU, Callable[..., _FDU]] = 0.9,
                 differential_weight: Union[_FDU, Callable[..., _FDU]] = 0.8,
                 evaluation=None,
                 replace_parents: bool = True,
                 replace_only_better: bool = False,
                 discard_parents: bool = False):
        """
        Differential evolution operator.  May be used as the whole evolution or as a single crossover operator.
        The parents fitness must be available.
        :param offsprings: Number of offsprings to create. May be float (then it is fraction of the original population
        to select), or integer (then it is number of individuals to select).
        :param crossover_probability: Probability that offspring inherit gene from donor vector.
        :param differential_weight: Direction strength parameter.
        :param evaluation: Evaluation pipe to evaluate new offsprings. Needed only if `replace_only_better` is true.
        :param replace_parents: Whether should offsprings replace their children (default) or concatenate themselves to the population.
        :param replace_only_better: Whether to replace parent only if offsprings is better. Default false.
        :param discard_parents: Whether to discard parents and return only offsprings. By default false.
        If set to true ignores it ignores both options above.
        """
        super().__init__(self.__handle_parent_fitnesses,
                         self.__report_offspring_fitness,
                         offsprings, crossover_probability,
                         differential_weight, evaluation,
                         replace_parents, replace_only_better, discard_parents)
        self.__fitnesses = None
        self.__parent_indices = None
        self.__children_fitness = None
        self.__picked_children = None

    def __handle_parent_fitnesses(self, _, indices, *args, **kwargs):
        return (self.__fitnesses[indices],), {}

    def __report_offspring_fitness(self, parent_indices=None, children_fitness = None, picked_children = None):
        self.__parent_indices = parent_indices
        self.__children_fitness = children_fitness
        self.__picked_children = picked_children

    def __call__(self, fitnesses, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply the differential operator on the population and return it.
        :param fitnesses: Population fitness in the same order as the population.
        :param population: Tensor representing the population.
        Expects the first dimension enumerate over the individuals.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Return fitness and mutated population.
        """

        self.__fitnesses = fitnesses

        (newpopulation, *arg), kargs = super().__call__(population, *args, **kwargs)

        if self._discard_parents:
            (children_f, *_), _ = self._evaluate(newpopulation)
            return (children_f, newpopulation, *args), kwargs
        elif self._replace_only_better:
            fitnesses[self.__parent_indices] = self.__children_fitness[self.__picked_children]
        elif not self._replace_parents:
            (children_f, *_), _ = self._evaluate(newpopulation[len(population):])
            fitnesses = t.cat([
                fitnesses,
                children_f
            ], dim = 0)
        else:
            (children_f, *_), _ = self._evaluate(newpopulation[self.__parent_indices])
            fitnesses[self.__parent_indices] = children_f

        return (fitnesses, newpopulation, *arg), kargs
