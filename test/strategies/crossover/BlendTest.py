###############################
#
# Created by Patrik Valkovic
# 02.04.21
#
###############################
import unittest
import torch as t
import ffeat.strategies.crossover as crossover
import ffeat
from test.repeat import repeat


class BlendTest(unittest.TestCase):
    def test_offsprings_absolute(self):
        s = crossover.Blend(40)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_fraction(self):
        s = crossover.Blend(0.4)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_absolute_without_replace(self):
        s = crossover.Blend(40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_fraction_without_replace(self):
        s = crossover.Blend(0.4, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_odd_offsprings(self):
        s = crossover.Blend(39)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100, 400))

    def test_odd_offsprings_replace(self):
        s = crossover.Blend(39, replace_parents=False)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (139, 400))

    def test_offsprings_absolute_not_inplace(self):
        s = crossover.Blend(40, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_fraction_absolute_not_inplace(self):
        s = crossover.Blend(0.4, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_more_dimensions(self):
        s = crossover.Blend(offsprings=40)
        pop = t.randn(100,17,13)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,17,13))

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.crossover.Blend(100, replace_parents=False),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40, device='cuda:0'),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.crossover.Blend(100, replace_parents=False),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
