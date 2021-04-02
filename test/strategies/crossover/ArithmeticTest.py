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
from ffeat.utils import decay
from test.repeat import repeat


class ArithmeticTest(unittest.TestCase):
    def test_offsprings_absolute(self):
        s = crossover.Arithmetic(40)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_fraction(self):
        s = crossover.Arithmetic(0.4)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_absolute_without_replace(self):
        s = crossover.Arithmetic(40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_fraction_without_replace(self):
        s = crossover.Arithmetic(0.4, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_odd_offsprings(self):
        s = crossover.Arithmetic(39)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100, 400))

    def test_odd_offsprings_replace(self):
        s = crossover.Arithmetic(39, replace_parents=False)
        pop = t.randn(100, 400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (139, 400))

    def test_offsprings_absolute_not_inplace(self):
        s = crossover.Arithmetic(40, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_fraction_absolute_not_inplace(self):
        s = crossover.Arithmetic(0.4, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_multiple_parents(self):
        s = crossover.Arithmetic(num_parents=7, offsprings=40)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))

    def test_custom_weight_float(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=1 / 5)
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_dimension(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((13,19), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_parent(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((5,), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_offspring(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((40,), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_offspringparent(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((40,5), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_offspringdimension(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((40,13,19), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_parentdimension(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((5,13,19), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_custom_weight_per_all(self):
        s = crossover.Arithmetic(num_parents=5, offsprings=40, parent_weight=t.full((40,5,13,19), 0.2))
        pop = t.randn(100,13,19)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,13,19))

    def test_multiple_parents_noreplace(self):
        s = crossover.Arithmetic(num_parents=7, offsprings=40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_more_dimensions(self):
        s = crossover.Arithmetic(offsprings=40)
        pop = t.randn(100,17,13)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,17,13))

    @unittest.skip('Not implemented yet')
    def test_absolute_offsprings_callback(self):
        s = crossover.Arithmetic(decay.Linear(60, 40, result_type=int))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    @unittest.skip('Not implemented yet')
    def test_fraction_offsprings_callback(self):
        s = crossover.Arithmetic(decay.Linear(0.6,0.4))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    def test_parents_callback(self):
        s = crossover.Arithmetic(offsprings=40, num_parents=decay.Linear(6, 2, result_type=int))
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc, iteration=14, max_iteration=100)
        self.assertEqual(newpop.shape, (100,400))

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Tournament(100),
            ffeat.strategies.mutation.AddFromNormal(0.01),
            ffeat.strategies.crossover.Arithmetic(80, replace_parents=False),
            iterations=1000
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 1))


if __name__ == '__main__':
    unittest.main()
