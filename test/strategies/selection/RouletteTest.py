###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import selection
from test.repeat import repeat


class RouletteTest(unittest.TestCase):
    def test_absolute(self):
        s = selection.Roulette(40)
        pop, fitness = t.rand((100,60)), t.rand((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (40,60))
        self.assertIsNot(newpop, pop)

    def test_fraction(self):
        s = selection.Roulette(0.4)
        pop, fitness = t.rand((100,60)), t.rand((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (40,60))
        self.assertIsNot(newpop, pop)

    def test_no_number(self):
        s = selection.Roulette()
        pop, fitness = t.rand((100,60)), t.rand((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertIsNot(newpop, pop)
        self.assertTrue(t.any(newpop != pop))

    def test_to_select_object(self):
        s = selection.Roulette(object())
        pop, fitness = t.rand((100,60)), t.rand((100,))
        with self.assertRaises(ValueError):
            s(pop, fitness)

    def test_is_better(self):
        s = selection.Roulette(1.0)
        old_pop = t.randn((1000,40))
        old_fitness = t.sum(t.pow(old_pop, 2.0), dim=-1)
        (new_pop,), kargs = s(t.clone(old_fitness), old_pop)
        new_fitness = t.sum(t.pow(new_pop, 2.0), dim=-1)
        self.assertGreater(t.mean(new_fitness), t.mean(old_fitness))

    def test_absolute_callback(self):
        s = selection.Roulette(ffeat.utils.decay.Linear(80, 40, result_type=int))
        pop, fitness = t.randn((100,60)), t.rand((100,))
        (newpop,), kargs = s(fitness, pop, iteration=30, max_iteration=40)
        self.assertEqual(newpop.shape, (50,60))
        self.assertIsNot(newpop, pop)

    def test_fraction_callback(self):
        s = selection.Roulette(ffeat.utils.decay.Linear(0.8, 0.4))
        pop, fitness = t.randn((100,60)), t.rand((100,))
        (newpop,), kargs = s(fitness, pop, iteration=30, max_iteration=40)
        self.assertEqual(newpop.shape, (50,60))
        self.assertIsNot(newpop, pop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_absolute_cuda(self):
        s = selection.Roulette(40)
        pop, fitness = t.randn((100,60), device='cuda:0'), t.rand((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_fraction_cuda(self):
        s = selection.Roulette(0.4)
        pop, fitness = t.randn((100,60), device='cuda:0'), t.rand((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_no_number_cuda(self):
        s = selection.Roulette()
        pop, fitness = t.rand((100,60), device='cuda:0'), t.rand((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    def test_negative_fitness(self):
        s = selection.Roulette(40)
        pop, fitness = t.rand((100,60)), t.rand((100,))
        fitness[31] = -1e-9
        with self.assertRaises(ValueError):
            s(fitness, pop)

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.utils.scaling.MultiplicativeInverse(),
            ffeat.utils.scaling.LogScale(0.1, 10.0),
            ffeat.strategies.selection.Roulette(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
            ffeat.strategies.crossover.OnePoint1D(0.4, replace_parents=True),
            iterations=1000
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
