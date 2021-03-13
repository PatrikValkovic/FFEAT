###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import unittest
import torch as t
import ffeat.measure as m


class FitnessMeanTest(unittest.TestCase):
    def test_should_execute(self):
        s = m.FitnessMean()
        f = t.randn((1000,))
        s(f)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_should_execute_cuda(self):
        s = m.FitnessMean()
        f = t.randn((1000,), device='cuda')
        s(f)

    def test_should_add_to_kwords(self):
        s = m.FitnessMean()
        f = t.randn((1000,))
        (nf,), k = s(f)
        self.assertIn("fitness_mean", k)
        self.assertEqual(k['fitness_mean'], float(t.mean(f)))
        self.assertIs(f, nf)


if __name__ == '__main__':
    unittest.main()
