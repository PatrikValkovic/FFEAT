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


class ArithmeticTest(unittest.TestCase):
    def test_offsprings_absolute(self):
        s = crossover.Arithmetic(num_offsprings=40)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_fraction(self):
        s = crossover.Arithmetic(fraction_offsprings=0.4)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_absolute_without_replace(self):
        s = crossover.Arithmetic(num_offsprings=40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_fraction_without_replace(self):
        s = crossover.Arithmetic(fraction_offsprings=0.4, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_odd_offsprings(self):
        s = crossover.Arithmetic(num_offsprings=39)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100, 400))

    def test_odd_offsprings_replace(self):
        s = crossover.Arithmetic(num_offsprings=39, replace_parents=False)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (139, 400))

    def test_offspring_count_not_set(self):
        with self.assertRaises(ValueError):
            crossover.Arithmetic()

    def test_offsprings_absolute_not_inplace(self):
        s = crossover.Arithmetic(num_offsprings=40, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_fraction_absolute_not_inplace(self):
        s = crossover.Arithmetic(fraction_offsprings=0.4, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_multiple_parents(self):
        s = crossover.Arithmetic(num_parents=7, num_offsprings=40)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))

    def test_custom_weight_float(self):
        s = crossover.Arithmetic(num_parents=5, num_offsprings=40, parent_weight=1/5)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))

    def test_custom_weight_dist(self):
        s = crossover.Arithmetic(num_parents=3, num_offsprings=40, parent_weight=t.distributions.Normal(1/3, 1))
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))

    def test_multiple_parents_noreplace(self):
        s = crossover.Arithmetic(num_parents=7, num_offsprings=40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_more_dimensions(self):
        s = crossover.Arithmetic(num_offsprings=40)
        pop = t.randn(100,17,13)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,17,13))

    def test_absolute_offsprings_callback(self):
        s = crossover.Arithmetic(num_offsprings=ffeat.decay.Linear(60,40,result_type=int))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    def test_fraction_offsprings_callback(self):
        s = crossover.Arithmetic(fraction_offsprings=ffeat.decay.Linear(0.6,0.4))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    def test_parents_callback(self):
        s = crossover.Arithmetic(num_offsprings=40, num_parents=ffeat.decay.Linear(6,2,result_type=int))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    def test_parents_weights_callback(self):
        class NormalStdCallback(ffeat.decay.Linear):
            def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs):
                val = super().__call__(*args, iteration=iteration, max_iteration=max_iteration, **kwargs)
                return t.distributions.Normal(0.5, val)
        s = crossover.Arithmetic(num_offsprings=40, parent_weight=NormalStdCallback(0.4, 0.2))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    def test_in_alg(self):
        class NormalStdCallback(ffeat.decay.Linear):
            def __call__(self, *args, iteration: int, max_iteration: int = None, **kwargs):
                val = super().__call__(*args, iteration=iteration, max_iteration=max_iteration, **kwargs)
                return t.distributions.Normal(0.5, val)
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.1, 0.1),
            ffeat.strategies.crossover.Arithmetic(40, num_parents=3, parent_weight=NormalStdCallback(0.5, 0.3)),
            iterations=500
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
