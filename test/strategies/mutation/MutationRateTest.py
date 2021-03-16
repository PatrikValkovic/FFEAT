###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import unittest
import torch as t
import ffeat
import ffeat.strategies.mutation as mut

class MutationRateTest(unittest.TestCase):
    def test_rate_decay(self):
        m = mut.AddFromNormal(0.1, ffeat.utils.decay.Linear(0.6, 0.1))
        pop = t.randn((1000, 400))
        (newpop,), kargs = m(pop, iteration=14, max_iteration=50)
        self.assertEqual(newpop.shape, pop.shape)
        self.assertIs(pop, newpop)

    def test_rate_decay_out_of_range(self):
        m = mut.AddFromNormal(0.1, ffeat.utils.decay.Linear(11.6, 8.1))
        pop = t.randn((1000, 400))
        with self.assertRaises(ValueError):
            m(pop, iteration=14, max_iteration=50)


if __name__ == '__main__':
    unittest.main()
