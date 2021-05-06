###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
"""
Module for Genetic Algorithms.
"""

from .Initialization import Initialization as initialization
from . import selection, crossover, mutation
from ffeat._common.Evaluation import EvalWrapper as evaluation
from ffeat._common.AlgorithmSkeleton import AlgorithmSkeleton as GeneticAlgorithm
