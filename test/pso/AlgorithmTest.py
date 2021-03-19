###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest
import torch as t
from ffeat import pso

class AlgorithmTest(unittest.TestCase):
    def test_PSO2006_random3(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    def test_PSO2006_circle2(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Circle(2),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    def test_PSO2006_grid_linear3(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('linear', 3, (10,10)),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    def test_PSO2006_grid_compact2(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('compact', 2, (10,10)),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    def test_PSO2006_grid_diamond2(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('diamond', 2, (10,10)),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    def test_PSO2006_grid_nearest10_work(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Nearest(10),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()

    def test_PSO2011_random3(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

if __name__ == '__main__':
    unittest.main()
