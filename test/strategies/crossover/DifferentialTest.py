###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import crossover, evaluation
from test.repeat import repeat


class DifferentialTest(unittest.TestCase):
    def test_absolute(self):
        d = crossover.Differential(600)
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(newpop, p)

    def test_fraction(self):
        d = crossover.Differential(0.6)
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(newpop, p)

    def test_absolute_no_replace(self):
        d = crossover.Differential(600, replace_parents=False)
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1600,400))

    @repeat(5)
    def test_absolute_discard(self):
        d = crossover.Differential(60, discard_parents=True)
        p = t.randn((100, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (60,400))
        self.assertIsNot(newpop, p)
        for i in range(100):
            for j in range(60):
                self.assertTrue(t.norm(p[i] - newpop[j]) > 1e-3)

    @repeat(5)
    def test_absolute_discard_with_fitness(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        d = crossover.DifferentialWithFitness(60, evaluation=evaluation.Evaluation(_f), discard_parents=True)
        p = t.randn((100, 400))
        f = _f(p)
        (nf, np), kargs = d(f, p)
        self.assertEqual(np.shape, (60,400))
        self.assertEqual(nf.shape, (60,))
        for i in range(100):
            for j in range(60):
                self.assertTrue(t.norm(p[i] - np[j]) > 1e-3)
        real_fitness = _f(np)
        self.assertTrue(t.all(t.abs(nf - real_fitness) < 1e-6))

    def test_fraction_no_replace(self):
        d = crossover.Differential(0.6, replace_parents=False)
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1600,400))

    def test_replace_better(self):
        fn = lambda x: t.sum(x ** 2, dim=-1)
        d = crossover.Differential(600, replace_only_better=True, evaluation=evaluation.Evaluation(fn))
        p = t.randn((1000, 400))
        (newpop,), kargs = d(t.clone(p))
        self.assertLessEqual(t.mean(fn(p)), t.mean(fn(newpop)))

    def test_replace_better_without_eval(self):
        with self.assertRaises(ValueError):
            crossover.Differential(600, replace_only_better=True)

    def test_CR_F_dist(self):
        d = crossover.Differential(600,
                                   crossover_probability=t.distributions.Uniform(0.7, 0.9),
                                   differential_weight=t.distributions.Uniform(0.75, 0.85))
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(newpop, p)

    def test_CR_F_callable(self):
        d = crossover.Differential(600,
                                   crossover_probability=lambda *_, **__: 0.9,
                                   differential_weight=lambda *_, **__: 0.7)
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(newpop, p)

    def test_CR_F_callable_dist(self):
        d = crossover.Differential(600,
                                   crossover_probability=lambda *_, **__: t.distributions.Uniform(0.7, 0.9),
                                   differential_weight=lambda *_, **__: t.distributions.Uniform(0.75, 0.85))
        p = t.randn((1000, 400))
        (newpop,), kargs = d(p)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(newpop, p)

    def test_with_fitnesses(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        d = crossover.DifferentialWithFitness(600, replace_only_better=True, evaluation=evaluation.Evaluation(_f))
        p = t.randn((1000, 400))
        f = _f(p)
        (nf, np), kargs = d(f, p)
        self.assertEqual(np.shape, (1000,400))
        self.assertIs(np, p)

    def test_with_fitnesses_replace_better_propagated(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        d = crossover.DifferentialWithFitness(600, replace_only_better=True, evaluation=evaluation.Evaluation(_f))
        p = t.randn((1000, 400))
        f = _f(p)
        (nf, np), kargs = d(f, p)
        real_fitness = _f(np)
        self.assertTrue(t.all(t.abs(nf - real_fitness) < 1e-6))

    def test_with_fitnesses_replace_propagated(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        d = crossover.DifferentialWithFitness(600, replace_parents=True, evaluation=evaluation.Evaluation(_f))
        p = t.randn((1000, 400))
        f = _f(p)
        (nf, np), kargs = d(f, p)
        real_fitness = _f(np)
        self.assertTrue(t.all(t.abs(nf - real_fitness) < 1e-6))

    def test_with_fitnesses_noreplace_propagated(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        d = crossover.DifferentialWithFitness(600, replace_parents=False, evaluation=evaluation.Evaluation(_f))
        p = t.randn((1000, 400))
        f = _f(p)
        (nf, np), kargs = d(f, p)
        real_fitness = _f(np)
        self.assertTrue(t.all(t.abs(nf - real_fitness) < 1e-6))

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.crossover.Differential(
                60,
                crossover_probability=lambda *_, **k: t.distributions.Normal(0.8, 0.1 * k['iteration'] / k['max_iteration']),
                differential_weight = lambda *_, **k: t.distributions.Normal(0.6, 0.1 * k['iteration'] / k['max_iteration'])
            ),
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
            ffeat.strategies.crossover.Differential(
                60,
                crossover_probability=lambda *_, **k: t.distributions.Normal(0.8, 0.1 * k['iteration'] / k['max_iteration']),
                differential_weight = lambda *_, **k: t.distributions.Normal(0.6, 0.1 * k['iteration'] / k['max_iteration'])
            ),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_cuda_replace_better(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40, device='cuda:0'),
            ffeat.strategies.crossover.Differential(
                0.6,
                crossover_probability=lambda *_, **k: t.distributions.Normal(0.8, 1.0 * k['iteration'] / k['max_iteration']),
                differential_weight=lambda *_, **k: t.distributions.Normal(0.3, 1.0 * k['iteration'] / k['max_iteration']),
                replace_only_better=True,
                evaluation=ffeat.strategies.evaluation.Evaluation(_f)
            ),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @repeat(5)
    def test_in_alg_replace_better_with_fitness(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.crossover.DifferentialWithFitness(
                100,
                replace_only_better=True,
                evaluation=ffeat.strategies.evaluation.Evaluation(_f)
            ),
            ffeat.strategies.selection.Tournament(100),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_replace_better_with_fitness_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -1.0, 1.0, 40, device='cuda:0'),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.crossover.DifferentialWithFitness(
                100,
                replace_only_better=True,
                evaluation=ffeat.strategies.evaluation.Evaluation(_f)
            ),
            ffeat.strategies.selection.Tournament(100),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
