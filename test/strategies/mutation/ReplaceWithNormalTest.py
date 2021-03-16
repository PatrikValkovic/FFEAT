###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.strategies import mutation


class ReplaceWithNormalTest(unittest.TestCase):
    def test_norm(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), 0.02)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(pop, newpop)

    def test_not_inplace(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), 0.02, in_place=False)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIsNot(pop, newpop)

    def test_rate_callable(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), ffeat.utils.decay.Linear(0.1, 0.01))
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop, iteration=13, max_iteration=23)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(pop, newpop)

    def test_rate_high(self):
        with self.assertRaises(ValueError):
            mutation.Replace(t.distributions.Normal(0.0, 5.0), 1.6)

    def test_rate_low(self):
        with self.assertRaises(ValueError):
            mutation.Replace(t.distributions.Normal(0.0, 5.0), 1.6)

    def test_rate_high_callable(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), ffeat.utils.decay.Linear(1.2, 1.4))
        pop = t.randn((1000,400))
        with self.assertRaises(ValueError):
            m(pop, iteration=13, max_iteration=23)

    def test_rate_low_callable(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), ffeat.utils.decay.Linear(-1.6, -0.2))
        pop = t.randn((1000,400))
        with self.assertRaises(ValueError):
            m(pop, iteration=13, max_iteration=23)

    def test_invalid_distribution_shape(self):
        m = mutation.Replace(t.distributions.Normal(0.0, t.ones((413,))), 0.02)
        pop = t.randn((1000,400))
        with self.assertRaises(ValueError):
            m(pop, iteration=13, max_iteration=23)


    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_cuda(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), 0.02)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(pop, newpop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_not_inplace_cuda(self):
        m = mutation.Replace(t.distributions.Normal(0.0, 5.0), 0.02, in_place=False)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIsNot(pop, newpop)


if __name__ == '__main__':
    unittest.main()
