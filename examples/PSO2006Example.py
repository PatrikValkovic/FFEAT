###############################
#
# Created by Patrik Valkovic
# 5/7/2021
#
###############################
import numpy as np
import ffeat.pso as pso
import ffeat.measure as measure
import bbobtorch

DIM = 40
problem = bbobtorch.create_f07(DIM)

best_fitness = []
mean_fitness = []

alg = pso.PSO(
    pso.initialization.Uniform(100, -5, 5, DIM), # position initialization
    pso.initialization.Uniform(100, -1, 1, DIM), # velocity initialization
    pso.evaluation.Evaluation(problem),
    pso.neighborhood.Random(3), # use Random neighborhood
    pso.update.PSO2006(), # use PSO2006 algorithm
    measurements_termination=[
        measure.FitnessLowest(measure.reporting.Array(best_fitness)),
        measure.FitnessMean(measure.reporting.Array(mean_fitness)),
    ],
    clip_position=pso.clip.Position(-5,5),
    iterations=100,
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), np.array(best_fitness) - float(problem.f_opt), label='Best fitness')
plt.plot(range(len(mean_fitness)), np.array(mean_fitness) - float(problem.f_opt), label='Mean fitness')
plt.legend()
plt.show()
