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
import ffeat.flow as flow


DIM = 40
problem = bbobtorch.create_f07(DIM)

best_fitness = []
mean_fitness = []

alg = ES.EvolutionStrategy(
    # initialization is sequence
    flow.Sequence(
        ES.initialization.Uniform(100, -5, 5, DIM),  # it allocates the population
        ES.evaluation.Evaluation(problem),  # and evaluate it at the beginning
    ),
    measure.FitnessLowest(measure.reporting.Array(best_fitness)),
    measure.FitnessMean(measure.reporting.Array(mean_fitness)),
    ES.crossover.DifferentialWithFitness(
        60, # create 60 offsprings
        evaluation=ES.evaluation.Evaluation(problem),
        replace_only_better=True, # replace parent only if the offsprign is better
    ),
    iterations=4000,
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), np.array(best_fitness) - float(problem.f_opt), label='Best fitness')
plt.plot(range(len(mean_fitness)), np.array(mean_fitness) - float(problem.f_opt), label='Mean fitness')
plt.legend()
plt.show()