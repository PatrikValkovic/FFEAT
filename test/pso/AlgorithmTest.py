###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest
import torch as t
from ffeat import pso
from test.repeat import repeat


class AlgorithmTest(unittest.TestCase):
    @repeat(5)
    def test_PSO2006_random3(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2006_circle4(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Circle(4),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2006_grid_linear3(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('linear', 3, (10,10)),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2006_grid_compact2(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('compact', 2, (10,10)),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2006_grid_diamond2(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Grid2D('diamond', 2, (10,10)),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2006_grid_nearest10_work(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Nearest(10),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
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

    @repeat(5)
    def test_PSO2006_float16(self):
        fn = lambda pop: t.sum(pop * pop, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40, dtype=t.float16),
            pso.initialization.Uniform(100,-1,1,40, dtype=t.float16),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(0.03),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 10.0))

    @repeat(5)
    def test_PSO2011_float16(self):
        fn = lambda pop: t.sum(pop * pop, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40, dtype=t.float16),
            pso.initialization.Uniform(100,-1,1,40, dtype=t.float16),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 10.0))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_PSO2006_random3_cuda(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40, device='cuda:0'),
            pso.initialization.Uniform(100,-1,1,40, device='cuda:0'),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))
        self.assertEqual(pop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_PSO2011_random3_cuda(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40, device='cuda:0'),
            pso.initialization.Uniform(100,-1,1,40, device='cuda:0'),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=0.8),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))
        self.assertEqual(pop.device, t.device('cuda:0'))

    @repeat(5)
    def test_PSO2006_random3_inertia_distribution(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=t.distributions.Normal(0.95, 0.01)),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @repeat(5)
    def test_PSO2011_random3_inertia_distribution(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(100,-5,5,40),
            pso.initialization.Uniform(100,-1,1,40),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Random(3),
            pso.update.PSO2011(local_c=0.2, global_c=0.3, inertia=t.distributions.Normal(0.8, 0.03)),
            iterations=500,
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))


if __name__ == '__main__':
    unittest.main()
