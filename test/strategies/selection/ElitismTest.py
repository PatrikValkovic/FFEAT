###############################
#
# Created by Patrik Valkovic
# 3/14/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import selection


class ElitismTest(unittest.TestCase):
    def test_should_work(self):
        s = selection.Elitism(10, selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))

    def test_should_keep_1(self):
        s = selection.Elitism(1, lambda *_, **__: ((t.rand((100,60)) + 10,), __))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertEqual(t.count_nonzero(t.all(newpop == pop, dim=-1)), 1)

    def test_should_keep_10(self):
        s = selection.Elitism(10, lambda *_, **__: ((t.rand((100,60)) + 10,), __))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertEqual(t.count_nonzero(t.all(newpop == pop, dim=-1)), 10)

    def test_should_keep_1_percentage(self):
        s = selection.Elitism(0.01, lambda *_, **__: ((t.rand((100,60)) + 10,), __))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertEqual(t.count_nonzero(t.all(newpop == pop, dim=-1)), 1)

    def test_should_keep_16_percentage(self):
        s = selection.Elitism(0.16, lambda *_, **__: ((t.rand((100,60)) + 10,), __))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertEqual(t.count_nonzero(t.all(newpop == pop, dim=-1)), 16)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_should_keep_16_percentage_cuda(self):
        s = selection.Elitism(0.16, lambda *_, **__: ((t.rand((100,60)) + 10,), __))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertEqual(t.count_nonzero(t.all(newpop == pop, dim=-1)), 16)

    def test_fraction_callback(self):
        s = selection.Elitism(ffeat.decay.Linear(0.1, 0.01), selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop, iteration=13, max_iteration=23)
        self.assertEqual(newpop.shape, (100,60))

    def test_absolute_callback(self):
        s = selection.Elitism(ffeat.decay.Linear(5, 2, result_type=int), selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop, iteration=13, max_iteration=23)
        self.assertEqual(newpop.shape, (100,60))

    def test_invalid_fraction(self):
        s = selection.Elitism(ffeat.decay.Linear(11.3, 6.2), selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        with self.assertRaises(ValueError):
            s(fitness, pop, iteration=13, max_iteration=23)

    def test_invalid_absolute(self):
        s = selection.Elitism(ffeat.decay.Linear(196, 112, result_type=int), selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        with self.assertRaises(ValueError):
            s(fitness, pop, iteration=13, max_iteration=23)

    def test_invalid_type_during_run(self):
        s = selection.Elitism(lambda *_, **__: object(), selection.Tournament())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        with self.assertRaises(ValueError):
            s(fitness, pop, iteration=13, max_iteration=23)

    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Elitism(2,
                ffeat.strategies.selection.Tournament(1.0),
                ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
                ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            ),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_in_alg_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40, device='cuda:0'),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Elitism(2,
                ffeat.strategies.selection.Tournament(1.0),
                ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
                ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            ),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))




if __name__ == '__main__':
    unittest.main()
