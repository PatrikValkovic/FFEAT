###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import unittest
import torch as t
import ffeat.measure as m



class QuantileTest(unittest.TestCase):
    def test_min(self):
        s = m.FitnessLowest()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_lowest', kargs)
        self.assertIn('fitness_q00', kargs)
        self.assertIn('fitness_q0.0', kargs)
        self.assertLess(abs(kargs['fitness_q0.0'] - float(t.min(f))), 1e-6)

    def test_max(self):
        s = m.FitnessHighest()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_highest', kargs)
        self.assertIn('fitness_q100', kargs)
        self.assertIn('fitness_q1.0', kargs)
        self.assertLess(abs(kargs['fitness_q1.0'] - float(t.max(f))), 1e-6)

    def test_median(self):
        s = m.FitnessMedian()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_median', kargs)
        self.assertIn('fitness_q50', kargs)
        self.assertIn('fitness_q0.5', kargs)
        self.assertLess(abs(kargs['fitness_q0.5'] - float(t.median(f))), 1e-2)

    def test_95(self):
        s = m.Fitness95Quantile()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_q95', kargs)
        self.assertIn('fitness_q0.95', kargs)
        self.assertLess(abs(kargs['fitness_q0.95'] - float(t.quantile(f,  0.95))), 1e-6)


    def test_99(self):
        s = m.Fitness99Quantile()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_q99', kargs)
        self.assertIn('fitness_q0.99', kargs)
        self.assertLess(abs(kargs['fitness_q0.99'] - float(t.quantile(f,  0.99))), 1e-6)

    def test_05(self):
        s = m.Fitness05Quantile()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_q05', kargs)
        self.assertIn('fitness_q0.05', kargs)
        self.assertLess(abs(kargs['fitness_q0.05'] - float(t.quantile(f,  0.05))), 1e-6)


    def test_01(self):
        s = m.Fitness01Quantile()
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_q01', kargs)
        self.assertIn('fitness_q0.01', kargs)
        self.assertLess(abs(kargs['fitness_q0.01'] - float(t.quantile(f,  0.01))), 1e-6)


    def test_34(self):
        s = m.FitnessQuantile(0.34)
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertIn('fitness_q34', kargs)
        self.assertIn('fitness_q0.34', kargs)
        self.assertLess(abs(kargs['fitness_q0.34'] - float(t.quantile(f, 0.34))), 1e-6)

    def test_0_613(self):
        s = m.FitnessQuantile(0.613)
        f = t.randn((1000,))
        (nf,), kargs = s(f)
        self.assertIs(nf, f)
        self.assertNotIn('fitness_q61', kargs)
        self.assertNotIn('fitness_q613', kargs)
        self.assertIn('fitness_q0.613', kargs)
        self.assertLess(abs(kargs['fitness_q0.613'] - float(t.quantile(f, 0.613))), 1e-6)


if __name__ == '__main__':
    unittest.main()
