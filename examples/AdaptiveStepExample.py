###############################
#
# Created by Patrik Valkovic
# 5/7/2021
#
###############################
import numpy as np
import ffeat.strategies as ES
import ffeat.measure as measure
import bbobtorch


DIM = 40
problem = bbobtorch.create_f07(DIM)

best_fitness = []
mean_fitness = []

alg = ES.EvolutionStrategy(
    ES.initialization.Uniform(100, -5, 5, DIM),
    ES.evaluation.Evaluation(problem),
    measure.FitnessLowest(measure.reporting.Array(best_fitness)),
    measure.FitnessMean(measure.reporting.Array(mean_fitness)),
    ES.mutation.AdaptiveStep(
        0.01, # starting deviation
        1.3, # standard deviation increase
        ES.evaluation.Evaluation(problem),
        std_decrease=0.2, # standard deviation decrease
        better_to_increase=0.3, # 30% individuals must be better than their parents to increase deviation
        minimum_std=0.00001, # minimum deviation
        maximum_std=1.0, # maximum deviation
    ),
    ES.selection.Tournament(100),  # Select 100 individuals
    ES.crossover.Uniform(0.6),  # Crossover 60% of them
    iterations=100,
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), np.array(best_fitness) - float(problem.f_opt), label='Best fitness')
plt.plot(range(len(mean_fitness)), np.array(mean_fitness) - float(problem.f_opt), label='Mean fitness')
plt.legend()
plt.show()