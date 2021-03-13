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
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(t.randn((100,40)))
        self.assertEqual(population.shape, (100,40))
        self.assertEqual(fitness.shape, (100,))

    def test_fitness_in_kwords(self):
        def _fn(population):
            return t.sum(population ** 2, dim=-1)
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(t.randn((100,40)))
        self.assertTrue('fitness' in kargs)
        self.assertIs(fitness, kargs['fitness'])

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_evaluate_cuda(self):
        def _fn(population):
            return t.sum(population ** 2, dim=-1)
        e = evaluation.Evaluation(_fn)
        (fitness, population), kargs = e(t.randn((100,40), device='cuda'))
        self.assertEqual(population.device, t.device('cuda:0'))
        self.assertEqual(fitness.device, t.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
