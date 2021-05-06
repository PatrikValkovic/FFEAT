###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
from typing import Union, Callable
import torch as t
from ffeat import Pipe, STANDARD_REPRESENTATION


class AddFromDistribution(Pipe):
    """
    Mutate population by adding value sampled from given distribution.
    """
    def __init__(self,
                 distribution: Union[t.distributions.Distribution, Callable[...,t.distributions.Distribution]],
                 mutation_rate: Union[float, Callable[..., float]] = 1.0,
                 in_place: bool = True
                 ):
        """
        Mutate population by adding value sampled from given distribution.
        :param distribution: Distribution from which to sample.
        :param mutation_rate: Probability of gene mutation. By default is 100%.
        :param in_place: Whether the operation should be done inplace.
        """
        if isinstance(mutation_rate, float) and (mutation_rate < 0.0 or mutation_rate > 1.0):
            raise ValueError("Mutation rate must be in the range [0.0, 1.0]")
        self._mutation_rate = self._handle_parameter(mutation_rate)
        self._distribution = self._handle_parameter(distribution)
        self.in_place = in_place

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply the operator and return mutated population.
        :param population: Population to mutate.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Mutated population.
        """
        dist = self._distribution(population, *args, **kwargs)
        mutation_rate = self._mutation_rate(population, *args, **kwargs)
        if mutation_rate < 0.0 or mutation_rate > 1.0:
            raise ValueError("Mutation rate must be in the range [0.0, 1.0]")
        modifications = dist.sample(sample_shape=population.shape).type(population.dtype).to(population.device)
        if mutation_rate < 1.0:
            to_modify = (t.rand(population.shape, device=population.device) < mutation_rate).type(t.int8)
            modifications = t.multiply(modifications, to_modify, out=modifications)
        population = t.add(population, modifications, out=population if self.in_place else modifications)
        return (population, *args), kwargs


class AddFromNormal(AddFromDistribution):
    """
    Mutate population by adding value sampled from normal distribution.
    """
    def __init__(self,
                 std: Union[float, Callable[..., float]],
                 mutation_rate: Union[float, Callable[..., float]] = 1.0,
                 in_place: bool = True):
        """
        Mutate population by adding value sampled from normal distribution.
        :param std: Standard deviation of the normal distribution.
        :param mutation_rate: Probability of gene mutation. By default is 100%.
        :param in_place: Whether the operation should be done inplace.
        """
        self._std = std
        dist = None if isinstance(std, Callable) else t.distributions.Normal(0.0, std)
        super().__init__(dist, mutation_rate, in_place)

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply the operator and return mutated population.
        :param population: Population to mutate.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Mutated population.
        """
        if isinstance(self._std, Callable):
            self._distribution = self._handle_parameter(t.distributions.Normal(0.0, self._std(population, *args, **kwargs)))
        return super().__call__(population, *args, **kwargs)


class AddFromCauchy(AddFromDistribution):
    """
    Mutate population by adding value sampled from Cauchy distribution.
    """
    def __init__(self,
                 scale: Union[float, Callable[..., float]],
                 mutation_rate: Union[float, Callable[..., float]] = 1.0,
                 in_place: bool = True):
        """
        Mutate population by adding value sampled from Cauchy distribution.
        :param scale: Scale of the cauchy distribution.
        :param mutation_rate: Probability of gene mutation. By default is 100%.
        :param in_place: Whether the operation should be done inplace.
        """
        self._scale = scale
        dist = None if isinstance(scale, Callable) else t.distributions.Cauchy(0.0, scale)
        super().__init__(dist, mutation_rate, in_place)

    def __call__(self, population, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Apply the operator and return mutated population.
        :param population: Population to mutate.
        :param args: Arguments passed along.
        :param kwargs: Keyword arguments passed along.
        :return: Mutated population.
        """
        if isinstance(self._scale, Callable):
            self._distribution = self._handle_parameter(t.distributions.Cauchy(0.0, self._scale(population, *args, **kwargs)))
        return super().__call__(population, *args, **kwargs)
