###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import mutation, evaluation
from test.repeat import repeat


class AddFromNormTest(unittest.TestCase):
    def test_one_step(self):
        fn = lambda x: t.sum(x ** 2, dim=-1)
        m = mutation.AdaptiveStep(0.5, 0.15, evaluation.Evaluation(fn))
        p = t.randn((1000, 400))
        f = fn(p)
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(newpop.shape, (1000,400))

    def test_add_offsprings(self):
        fn = lambda x: t.sum(x ** 2, dim=-1)
        m = mutation.AdaptiveStep(0.5, 0.15, evaluation.Evaluation(fn), replace_parents=False)
        p = t.randn((1000, 400))
        f = fn(p)
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(newpop.shape, (2000,400))

    def test_specify_offsprings_fraction(self):
        fn = lambda x: t.sum(x ** 2, dim=-1)
        m = mutation.AdaptiveStep(0.5, 0.15, evaluation.Evaluation(fn), replace_parents=False, mutate_members=0.6)
        p = t.randn((1000, 400))
        f = fn(p)
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(newpop.shape, (1600,400))

    def test_specify_offsprings_absolute(self):
        fn = lambda x: t.sum(x ** 2, dim=-1)
        m = mutation.AdaptiveStep(0.5, 0.15, evaluation.Evaluation(fn), replace_parents=False, mutate_members=200)
        p = t.randn((1000, 400))
        f = fn(p)
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(newpop.shape, (1200,400))

    def test_increase_stepsize(self):
        fn = lambda x: t.zeros((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn))
        p = t.randn((1000, 400))
        f = t.ones((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertGreater(m._current_std, 0.5)
        self.assertEqual(m._current_std, 0.5 * 1.5)

    def test_decrease_stepsize(self):
        fn = lambda x: t.ones((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn))
        p = t.randn((1000, 400))
        f = t.zeros((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertLess(m._current_std, 0.5)
        self.assertEqual(m._current_std, 0.5 / 1.5)

    def test_decrease_specified(self):
        fn = lambda x: t.ones((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn), std_decrease=0.99)
        p = t.randn((1000, 400))
        f = t.zeros((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertLess(m._current_std, 0.5)
        self.assertEqual(m._current_std, 0.5 * 0.99)

    def test_decrease_clip(self):
        fn = lambda x: t.ones((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn), minimum_std=0.4)
        p = t.randn((1000, 400))
        f = t.zeros((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(m._current_std, 0.4)

    def test_increase_clip(self):
        fn = lambda x: t.zeros((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn), maximum_std=0.6)
        p = t.randn((1000, 400))
        f = t.ones((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(m._current_std, 0.6)

    def test_decrease_becase_not_enough(self):
        fn = lambda x: t.zeros((1000,))
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(fn), better_to_increase=2000)
        p = t.randn((1000, 400))
        f = t.ones((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(m._current_std, 0.5 / 1.5)

    def test_replace_only_better(self):
        better_indices = t.unique(t.randint(1000, (314,), dtype=t.long))
        new_f = t.ones((1000,))
        new_f[better_indices] = 0
        def _fn(_):
            nonlocal new_f
            return new_f
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(_fn), replace_only_better=True)
        p = t.randn((1000, 400))
        old_pop = t.clone(p)
        f = t.ones((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertEqual(t.sum(newfitness == 0), len(better_indices))
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), len(better_indices))

    def test_replace_only_better_fraction(self):
        better_indices = t.unique(t.randint(800, (314,), dtype=t.long))
        new_f = t.ones((800,))
        new_f[better_indices] = 0
        def _fn(_):
            nonlocal new_f
            return new_f
        m = mutation.AdaptiveStep(0.5, 1.5, evaluation.Evaluation(_fn), replace_only_better=True, mutate_members=800)
        p = t.randn((1000, 400))
        old_pop = t.clone(p)
        f = t.ones((1000,))
        (newfitness, newpop), kargs = m(f, p)
        self.assertLess(t.sum(newfitness == 0), len(better_indices))
        self.assertLess(t.sum(t.any(newpop != old_pop, dim=-1)), len(better_indices))

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.mutation.AdaptiveStep(
                0.4, 1.5, ffeat.strategies.evaluation.Evaluation(_f),
                replace_parents=False,
                mutate_members=60,
                minimum_std=0.001, maximum_std=0.9
            ),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.mutation.AdaptiveStep(
                0.4, 1.5, ffeat.strategies.evaluation.Evaluation(_f),
                replace_parents=False,
                mutate_members=60,
                minimum_std=1e-12, maximum_std=0.9
            ),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
