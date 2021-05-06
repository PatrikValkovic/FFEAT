###############################
#
# Created by Patrik Valkovic
# 4/17/2021
#
###############################
"""
Module implementing parental sampling strategies for various crossover operators.
"""
import torch as t



def randint(popsize, samples, parents_per_sample, device):
    """
    Samples parents using the `torch.randint` call.
    The parents may be therefore duplicated, but is faster than multinomial sampling.
    :param popsize: Size of the original population, also number of potential parents.
    :param samples: How many samples to create, should be equal to number of offsprings.
    :param parents_per_sample: How many parents each sample have, is equal to number of parents per one offspring.
    :param device: Device on which to sample.
    :return: Indices of the parents as `torch.long` datatype in the shape (samples, parents_per_sample).
    """
    return t.randint(popsize, (samples, parents_per_sample), device=device, dtype=t.long)


def multinomial(popsize, samples, parents_per_sample, device):
    """
    Samples parents using the `torch.multinomial` call.
    The parents of one offspring is not duplicated, but is by order of magnitude slower than `randint` sampling.
    :param popsize: Size of the original population, also number of potential parents.
    :param samples: How many samples to create, should be equal to number of offsprings.
    :param parents_per_sample: How many parents each sample have, is equal to number of parents per one offspring.
    The parents along this dimension are unique.
    :param device: Device on which to sample.
    :return: Indices of the parents as `torch.long` datatype in the shape (samples, parents_per_sample).
    """
    probs = t.tensor(1 / popsize, device=device, dtype=t.float)
    probs.as_strided_((samples,popsize), (0,0))
    sample = t.multinomial(probs, parents_per_sample, replacement=False)
    return sample
