###############################
#
# Created by Patrik Valkovic
# 5/7/2021
#
###############################
import ffeat.genetic as GA
import ffeat.measure as measure
from SATProblem import SATProblem
from ffeat.utils import decay

FILE = "uf250-017.cnf"
problem = SATProblem.from_cnf_file(FILE)

best_fitness = []
mean_fitness = []

alg = GA.GeneticAlgorithm(
    GA.initialization.Uniform(100, problem.nvars),
    GA.evaluation.Evaluation(problem.fitness_count_unsatisfied),
    measure.FitnessLowest(measure.reporting.Array(best_fitness)),
    measure.FitnessMean(measure.reporting.Array(mean_fitness)),
    GA.selection.Tournament(100),  # Select 100 individuals
    GA.crossover.TwoPoint1D(0.6),  # Crossover 60% of them
    # Mutate 40% of them with probability change decreasing polynomially from 0.01 to 0.0005
    GA.mutation.FlipBit(0.4, decay.Polynomial(0.01, 0.0005, 3)),
    iterations=500,
)
alg()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(best_fitness)), best_fitness, label='Best fitness')
plt.plot(range(len(mean_fitness)), mean_fitness, label='Mean fitness')
plt.legend()
plt.show()
