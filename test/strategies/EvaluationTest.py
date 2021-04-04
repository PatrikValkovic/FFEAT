###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import unittest
import torch as t
from ffeat.strategies import evaluation


class EvaluationTest(unittest.TestCase):
    def test_should_call(self):
        called = False
        def _fn(population):
            nonlocal called
            called = True
        e = evaluation.Evaluation(_fn)
        e(None)
        self.assertTrue(called)

    def test_should_evaluate(self):
        def _fn(population):
            return t.sum(population ** 2, dim=-1)
        pop = t.randn((100,40))
        real_fitness = _fn(pop)
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(pop)
        self.assertEqual(population.shape, (100,40))
        self.assertEqual(fitness.shape, (100,))
        self.assertTrue(t.all(t.abs(real_fitness - fitness)) < 1e-6)

    def test_fitness_in_kwords(self):
        def _fn(population):
            return t.sum(population ** 2, dim=-1)
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(t.randn((100,40)))
        self.assertIn('orig_fitness', kargs)
        self.assertIs(fitness, kargs['orig_fitness'])

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_evaluate_cuda(self):
        def _fn(population):
            return t.sum(population ** 2, dim=-1)
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(t.randn((100,40), device='cuda'))
        self.assertEqual(population.device, t.device('cuda:0'))
        self.assertEqual(fitness.device, t.device('cuda:0'))

    def test_should_evaluate_rowlike(self):
        def _fn(member):
            return t.sum(member ** 2)
        pop = t.randn((100,40))
        real_fitness = t.sum(pop ** 2, dim = -1)
        e = evaluation.RowEval(_fn)
        (fitness, population), kargs = e(pop)
        self.assertEqual(population.shape, (100,40))
        self.assertEqual(fitness.shape, (100,))
        self.assertTrue(t.all(t.abs(real_fitness - fitness)) < 1e-6)

    def test_rowlike_should_be_called_multiple_times(self):
        called = 0
        def _fn(_):
            nonlocal called
            called += 1
            return t.tensor(0)
        e = evaluation.RowEval(_fn)
        _ = e(t.randn(1000,400))
        self.assertEqual(called, 1000)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_rowlike_cuda(self):
        def _fn(member):
            return t.sum(member ** 2)
        pop = t.randn((100,40), device='cuda:0')
        real_fitness = t.sum(pop ** 2, dim = -1)
        e = evaluation.RowEval(_fn)
        (fitness, population), kargs = e(pop)
        self.assertEqual(population.shape, (100,40))
        self.assertEqual(fitness.shape, (100,))
        self.assertTrue(t.all(t.abs(real_fitness - fitness)) < 1e-6)
        self.assertEqual(fitness.device, t.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
