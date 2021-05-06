###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
"""
Module that allows to measure and log fitness statistics.
"""
from .FitnessQuantile import FitnessMedian, FitnessQuantile, Fitness95Quantile, Fitness99Quantile, Fitness05Quantile, Fitness01Quantile
from . import reporting
from .fitness_highest_lowest import FitnessLowest, FitnessHighest
from .FitnessStd import FitnessStd
from .FitnessMean import FitnessMean
