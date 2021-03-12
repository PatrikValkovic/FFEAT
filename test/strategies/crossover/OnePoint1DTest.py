###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import unittest
from pydoc import cram

import torch as t
from ffeat.strategies import crossover


class OnePoint1DTest(unittest.TestCase):
    def test_offsprings_absolute(self):
        s = crossover.OnePoint1D(num_offsprings=40)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_offsprings_fraction(self):
        s = crossover.OnePoint1D(crossover_percentage=0.4)
        pop = t.randn(100,400)
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIs(popc, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_absolute_without_replace(self):
        s = crossover.OnePoint1D(num_offsprings=40, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_fraction_without_replace(self):
        s = crossover.OnePoint1D(crossover_percentage=0.4, replace_parents=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (140,400))

    def test_odd_offsprings(self):
        with self.assertRaises(ValueError):
            crossover.OnePoint1D(num_offsprings=39)

    def test_offspring_count_not_set(self):
        with self.assertRaises(ValueError):
            crossover.OnePoint1D()

    def test_offsprings_absolute_not_inplace(self):
        s = crossover.OnePoint1D(num_offsprings=40, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    def test_fraction_absolute_not_inplace(self):
        s = crossover.OnePoint1D(crossover_percentage=0.4, in_place=False)
        pop = t.randn(100,400)
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.shape, (100,400))
        self.assertIsNot(pop, newpop)
        self.assertLess(t.sum(t.any(pop != newpop, dim=-1)), 40)
        self.assertGreaterEqual(t.sum(t.any(pop == newpop, dim=-1)), 60)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_absolute_cuda(self):
        s = crossover.OnePoint1D(num_offsprings=40)
        pop = t.randn(100,400, device='cuda:0')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_fraction_cuda(self):
        s = crossover.OnePoint1D(crossover_percentage=0.4)
        pop = t.randn(100,400, device='cuda:0')
        popc = t.clone(pop)
        (newpop,), kargs = s(popc)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_offsprings_absolute_not_inplace_cuda(self):
        s = crossover.OnePoint1D(num_offsprings=40, in_place=False)
        pop = t.randn(100,400, device='cuda:0')
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_fraction_absolute_not_inplace_cuda(self):
        s = crossover.OnePoint1D(crossover_percentage=0.4, in_place=False)
        pop = t.randn(100,400, device='cuda:0')
        (newpop,), kargs = s(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
