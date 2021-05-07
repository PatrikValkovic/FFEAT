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
    ES.selection.Tournament(100),  # Select 100 individuals
    ES.crossover.Blend(0.6, alpha=0.5),  # Crossover 60% of them
    ES.mutation.AddFromNormal(0.001, 1.0),  # Normal mutate all population with standard deviation 0.001
    iterations=100,
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), np.array(best_fitness) - float(problem.f_opt), label='Best fitness')
plt.plot(range(len(mean_fitness)), np.array(mean_fitness) - float(problem.f_opt), label='Mean fitness')
plt.legend()
plt.show()