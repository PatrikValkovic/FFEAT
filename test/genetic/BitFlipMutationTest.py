###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat._common.initialization.Uniform import UniformInit
from ffeat.genetic import mutation
from test.repeat import repeat


class BitFlipMutationTest(unittest.TestCase):
    def test_abs_inplace(self):
        m = mutation.FlipBit(60, 0.05)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, old_pop.shape)
        self.assertIs(pop, newpop)
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), 60)

    def test_fraction_inplace(self):
        m = mutation.FlipBit(0.6, 0.05)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, old_pop.shape)
        self.assertIs(pop, newpop)
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), 60)

    def test_abs(self):
        m = mutation.FlipBit(60, 0.05, in_place=False)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (160,400))
        self.assertIsNot(pop, newpop)

    def test_fraction(self):
        m = mutation.FlipBit(0.6, 0.05, in_place=False)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (160,400))
        self.assertIsNot(pop, newpop)

    def test_more_dimension(self):
        m = mutation.FlipBit(0.6, 0.05)
        pop = (t.rand((100, 200, 60, 30)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (100, 200, 60, 30))
        self.assertIs(pop, newpop)
        self.assertFalse(t.all(old_pop == newpop))

    def test_probability_high(self):
        with self.assertRaises(ValueError):
            mutation.FlipBit(40, 1 + 1e-7)

    def test_probability_low(self):
        with self.assertRaises(ValueError):
            mutation.FlipBit(40, -1e-7)

    def test_prob_callback(self):
        m = mutation.FlipBit(60, lambda *_, **__: 0.05)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, old_pop.shape)
        self.assertIs(pop, newpop)
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), 60)

    def test_num_callback_integer(self):
        m = mutation.FlipBit(lambda *_, **__: 60, 0.05)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, old_pop.shape)
        self.assertIs(pop, newpop)
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), 60)

    def test_num_callback_fraction(self):
        m = mutation.FlipBit(lambda *_, **__: 0.6, 0.05)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        old_pop = t.clone(pop)
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, old_pop.shape)
        self.assertIs(pop, newpop)
        self.assertEqual(t.sum(t.any(newpop != old_pop, dim=-1)), 60)

    def test_num_callback_low(self):
        m = mutation.FlipBit(60, lambda *_, **__: -1e-7)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        with self.assertRaises(ValueError):
            m(pop)

    def test_num_callback_high(self):
        m = mutation.FlipBit(60, lambda *_, **__: 1+1e-7)
        pop = (t.rand((100, 400)) < 0.4).to(t.uint8)
        with self.assertRaises(ValueError):
            m(pop)

    @repeat(5)
    def test_in_alg_unsigned(self):
        fn = lambda x: t.sum(x, dim=-1, dtype=t.int32)
        a = ffeat.genetic.GeneticAlgorithm(
            ffeat.genetic.initialization.Uniform(100, 40),
            ffeat.genetic.evaluation.Evaluation(fn),
            ffeat.genetic.selection.Tournament(100),
            ffeat.genetic.mutation.FlipBit(0.6, mutate_prob=0.01),
        )
        (pop,), kargs = a()
        self.assertTrue(t.all(fn(pop) < 5))

    @repeat(5)
    def test_in_alg_signed(self):
        fn = lambda x: t.sum(x, dim=-1, dtype=t.int32)
        a = ffeat.genetic.GeneticAlgorithm(
            UniformInit(100, 0, 2, 40, t.int16),
            ffeat.genetic.evaluation.Evaluation(fn),
            ffeat.genetic.selection.Tournament(100),
            ffeat.genetic.mutation.FlipBit(0.6, mutate_prob=0.01),
        )
        (pop,), kargs = a()
        self.assertTrue(t.all(fn(pop) < 5))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_unsigned_cuda(self):
        fn = lambda x: t.sum(x, dim=-1, dtype=t.int32)
        a = ffeat.genetic.GeneticAlgorithm(
            ffeat.genetic.initialization.Uniform(100, 40, device='cuda:0'),
            ffeat.genetic.evaluation.Evaluation(fn),
            ffeat.genetic.selection.Tournament(100),
            ffeat.genetic.mutation.FlipBit(0.6, mutate_prob=0.01),
        )
        (pop,), kargs = a()
        self.assertTrue(t.all(fn(pop) < 5))
        self.assertEqual(pop.device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_signed_cuda(self):
        fn = lambda x: t.sum(x, dim=-1, dtype=t.int32)
        a = ffeat.genetic.GeneticAlgorithm(
            UniformInit(100, 0, 2, 40, t.int16, device='cuda:0'),
            ffeat.genetic.evaluation.Evaluation(fn),
            ffeat.genetic.selection.Tournament(100),
            ffeat.genetic.mutation.FlipBit(0.6, mutate_prob=0.01),
        )
        (pop,), kargs = a()
        self.assertTrue(t.all(fn(pop) < 5))
        self.assertEqual(pop.device, t.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
