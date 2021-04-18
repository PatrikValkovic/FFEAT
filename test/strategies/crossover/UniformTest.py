###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import crossover
from test.repeat import repeat


class UniformTest(unittest.TestCase):
    def test_offsprings_absolute(self):
        s = crossover.Uniform(40)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_fraction(self):
        s = crossover.Uniform(0.4)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_absolute_without_replace(self):
        s = crossover.Uniform(40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_fraction_without_replace(self):
        s = crossover.Uniform(0.4, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_odd_offsprings(self):
        with self.assertRaises(ValueError):
            crossover.Uniform(39)

    def test_offsprings_absolute_not_inplace(self):
        s = crossover.Uniform(40, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_discard_parents(self):
        s = crossover.Uniform(40, discard_parents=True)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (40,400))
        self.assertIsNot(pop, newpop)


    def test_fraction_absolute_not_inplace(self):
        s = crossover.Uniform(0.4, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_absolute_cuda(self):
        s = crossover.Uniform(40)
        pop = t.randn(100,400, device='cuda:0')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_fraction_cuda(self):
        s = crossover.Uniform(0.4)
        pop = t.randn(100,400, device='cuda:0')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_absolute_not_inplace_cuda(self):
        s = crossover.Uniform(40, in_place=False)
        pop = t.randn(100,400, device='cuda:0')
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_fraction_absolute_not_inplace_cuda(self):
        s = crossover.Uniform(0.4, in_place=False)
        pop = t.randn(100,400, device='cuda:0')
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    def test_multidimensional(self):
        s = crossover.Uniform(40)
        pop = t.randn(100,40,10,50)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,40,10,50))
        self.assertIs(popc, newpop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_multidimensional_cuda(self):
        s = crossover.Uniform(40)
        pop = t.randn(100,40,10,50, device='cuda:0')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,40,10,50))
        self.assertIs(popc, newpop)

    def test_multidimensional_not_inplace(self):
        s = crossover.Uniform(40, replace_parents=False)
        pop = t.randn((100,40,10,50))
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (140,40,10,50))
        self.assertIsNot(popc, pop)
        self.assertTrue(t.all(popc == pop))

    def test_mutation_rate(self):
        s = crossover.Uniform(40, change_prob=0.1)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
            ffeat.strategies.crossover.Uniform(40, replace_parents=True, change_prob=0.3),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    def test_parental_sampling(self):
        s = crossover.Uniform(40, parental_sampling=ffeat.utils.parental_sampling.multinomial)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_parental_sampling_cuda(self):
        s = crossover.Uniform(40, parental_sampling=ffeat.utils.parental_sampling.multinomial)
        pop = t.randn(100,400, device='cuda')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)


if __name__ == '__main__':
    unittest.main()
