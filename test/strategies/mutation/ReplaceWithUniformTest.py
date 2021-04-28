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
from test.repeat import repeat


class ReplaceWithUniformTest(unittest.TestCase):
    def test_float(self):
        m = mutation.ReplaceUniform(-5.0, 5.0, 0.02)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(pop, newpop)

    def test_int(self):
        m = mutation.ReplaceUniform(-5, 5, 0.02)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIs(pop, newpop)

    def test_out_of_place(self):
        m = mutation.ReplaceUniform(-5.0, 5.0, 0.02, in_place=False)
        pop = t.randn((1000,400))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,400))
        self.assertIsNot(pop, newpop)

    def test_list_float(self):
        m = mutation.ReplaceUniform([-5.0]*7, [5.0]*7, 0.02)
        pop = t.randn((1000,7))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)

    def test_list_int(self):
        m = mutation.ReplaceUniform([-5]*7, [5]*7, 0.02)
        pop = t.randn((1000,7))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)

    def test_tensor(self):
        m = mutation.ReplaceUniform(
            t.full((7,), -5.0, dtype=t.float32, device='cpu'),
            t.full((7,), 5.0, dtype=t.float32, device='cpu'),
            0.02
        )
        pop = t.randn((1000,7))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)

    def test_min_tensor(self):
        m = mutation.ReplaceUniform(
            t.full((7,), -5.0, dtype=t.float32, device='cpu'),
            5.0,
            0.02
        )
        pop = t.randn((1000,7))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)


    def test_max_tensor(self):
        m = mutation.ReplaceUniform(
            -5.0,
            t.full((7,), 5.0, dtype=t.float32, device='cpu'),
            0.02
        )
        pop = t.randn((1000,7))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)


    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_max_cuda(self):
        m = mutation.ReplaceUniform(
            -5.0,
            t.full((7,), 5.0, dtype=t.float16, device='cuda:0'),
            0.02
        )
        pop = t.randn((1000,7), device=t.device('cuda:0'))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_min_cuda(self):
        m = mutation.ReplaceUniform(
            t.full((7,), -5.0, dtype=t.float16, device='cuda:0'),
            5.0,
            0.02
        )
        pop = t.randn((1000,7), device=t.device('cuda:0'))
        (newpop,), kargs = m(pop)
        self.assertEqual(newpop.shape, (1000,7))
        self.assertIs(pop, newpop)

    @repeat(5)
    def test_in_alg(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Elitism(40,
               ffeat.strategies.selection.Tournament(100),
               ffeat.strategies.crossover.OnePoint1D(80, replace_parents=False),
               ffeat.strategies.mutation.ReplaceUniform(-1, 1, 0.01),
            ),
            iterations=1000
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 2))

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    @repeat(5)
    def test_in_alg_cuda(self):
        _f = lambda x: t.sum(t.pow(x, 2), dim=-1)
        alg = ffeat.strategies.EvolutionStrategy(
            ffeat.strategies.initialization.Uniform(100, -5.0, 5.0, 40, device='cuda'),
            ffeat.strategies.evaluation.Evaluation(_f),
            ffeat.strategies.selection.Elitism(40,
               ffeat.strategies.selection.Tournament(100),
               ffeat.strategies.crossover.OnePoint1D(80, replace_parents=False),
               ffeat.strategies.mutation.ReplaceUniform(-1, 1, 0.01),
            ),
            iterations=1000
        )
        (pop,), kargs = alg()
        self.assertTrue(t.all(_f(pop) < 2))


if __name__ == '__main__':
    unittest.main()
