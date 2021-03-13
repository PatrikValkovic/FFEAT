###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import unittest
import torch as t
import ffeat


class StrategyTest(unittest.TestCase):
    def test_simple(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
            ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_simple_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40, device='cuda:0'),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
            ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
