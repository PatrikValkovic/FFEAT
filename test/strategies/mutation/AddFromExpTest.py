###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import math
import unittest
import torch as t
from ffeat.strategies import mutation


class AddFromNormTest(unittest.TestCase):
    def test_from_exp(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0))
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, pop.shape)
        self.assertIs(pop, newpop)

    def test_smaller_mutation_rate(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0), 0.5)
        pop = t.randn((1000,1000))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, pop.shape)
        self.assertIs(pop, newpop)

    def test_not_in_place(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0), in_place=False)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, pop.shape)
        self.assertIsNot(pop, newpop)

    def test_some_unchanged(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0), 0.5, in_place=False)
        pop = t.randn((1000,1000))
        (newpop,), kargs = m(pop)
        self.assertTrue(t.any(newpop == pop))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_inplace_cuda(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0), 0.5)
        pop = t.randn((1000,1000), device='cuda:0')
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))
        self.assertIs(newpop, pop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_not_inplace_cuda(self):
        m = mutation.AddFromDistribution(t.distributions.Exponential(1.0), 0.5, in_place=False)
        pop = t.randn((1000,1000), device='cuda:0')
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))
        self.assertIsNot(newpop, pop)

    def test_negative_mutation_rate(self):
        with self.assertRaises(ValueError):
             mutation.AddFromDistribution(t.distributions.Exponential(1.0), -0.1)

    def test_big_mutation_rate(self):
        with self.assertRaises(ValueError):
             d = mutation.AddFromDistribution(t.distributions.Exponential(1.0), 1.0001)

    def test_inf_mutation_rate(self):
        with self.assertRaises(ValueError):
             mutation.AddFromDistribution(t.distributions.Exponential(1.0), math.inf)


if __name__ == '__main__':
    unittest.main()
