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

class TournamentTest(unittest.TestCase):
    def test_absolute(self):
        s = selection.Tournament(40)
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (40,60))
        self.assertIsNot(newpop, pop)

    def test_fraction(self):
        s = selection.Tournament(0.4)
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (40,60))
        self.assertIsNot(newpop, pop)

    def test_no_number(self):
        s = selection.Tournament()
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.shape, (100,60))
        self.assertIsNot(newpop, pop)
        self.assertTrue(t.any(newpop != pop))

    def test_to_select_object(self):
        s = selection.Tournament(object())
        pop, fitness = t.rand((100,60)), t.randn((100,))
        with self.assertRaises(ValueError):
            s(pop, fitness)

    def test_is_better(self):
        s = selection.Tournament(1.0)
        old_pop = t.randn((1000,40))
        old_fitness = t.sum(t.pow(old_pop, 2.0), dim=-1)
        (new_pop,), kargs = s(old_fitness, old_pop)
        new_fitness = t.sum(t.pow(new_pop, 2.0), dim=-1)
        self.assertLess(t.mean(new_fitness), t.mean(old_fitness))

    def test_absolute_callback(self):
        s = selection.Tournament(ffeat.decay.Linear(80,40, result_type=int))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop, iteration=30, max_iteration=40)
        self.assertEqual(newpop.shape, (50,60))
        self.assertIsNot(newpop, pop)

    def test_fraction_callback(self):
        s = selection.Tournament(ffeat.decay.Linear(0.8, 0.4))
        pop, fitness = t.rand((100,60)), t.randn((100,))
        (newpop,), kargs = s(fitness, pop, iteration=30, max_iteration=40)
        self.assertEqual(newpop.shape, (50,60))
        self.assertIsNot(newpop, pop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_absolute_cuda(self):
        s = selection.Tournament(40)
        pop, fitness = t.rand((100,60), device='cuda:0'), t.randn((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_fraction_cuda(self):
        s = selection.Tournament(0.4)
        pop, fitness = t.rand((100,60), device='cuda:0'), t.randn((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_no_number_cuda(self):
        s = selection.Tournament()
        pop, fitness = t.rand((100,60), device='cuda:0'), t.randn((100,), device='cuda:0')
        (newpop,), kargs = s(fitness, pop)
        self.assertEqual(newpop.device, t.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
