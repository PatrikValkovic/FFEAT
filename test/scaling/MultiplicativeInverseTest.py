###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import unittest
import torch as t
from ffeat.utils.scaling import MultiplicativeInverse

class MultiplicativeInverseTest(unittest.TestCase):
    def test_for_float(self):
        inv = MultiplicativeInverse()
        f = t.rand((1000,))+1e-9
        pf = t.clone(f)
        (nf,), kargs = inv(pf)
        self.assertIs(nf, pf)
        for expected, actual in zip(1.0 / f, nf):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_for_int(self):
        inv = MultiplicativeInverse()
        f = (t.rand((1000,)) * 875).type(t.int32) + 1
        pf = t.clone(f)
        (nf,), kargs = inv(pf)
        for expected, actual in zip(1.0 / f, nf):
            self.assertLess(abs(expected - actual), 1e-6)


if __name__ == '__main__':
    unittest.main()
