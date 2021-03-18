###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat import pso


class SimpleAlgTest(unittest.TestCase):
    def test_simple_alg(self):
        fn = lambda pop: t.sum(pop ** 2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(11*13,-5,5,40),
            pso.initialization.Uniform(11*13,-1,1,40),
            pso.evaluation.Evaluation(fn),
            #pso.neighborhood.Random(35),
            #pso.neighborhood.Static(pso.neighborhood.Random(35)),
            pso.neighborhood.Nearest(35),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_simple_alg_cuda(self):
        fn = lambda pop: t.sum(pop **2, dim=1)
        alg = pso.PSO(
            pso.initialization.Uniform(11*13,-5,5,40, device='cuda:0'),
            pso.initialization.Uniform(11*13,-1,1,40, device='cuda:0'),
            pso.evaluation.Evaluation(fn),
            pso.neighborhood.Nearest(35),
            pso.update.PSO2006(local_c=0.2, global_c=0.3, inertia=0.95),
            iterations=500
        )
        (pop,), kwargs = alg()
        self.assertTrue(t.all(fn(pop) < 1.0))


if __name__ == '__main__':
    unittest.main()
