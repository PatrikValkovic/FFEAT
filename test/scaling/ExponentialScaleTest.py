###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import unittest
import torch as t
from ffeat.utils import scaling


class ExponentialScaleTest(unittest.TestCase):
    def test_zero_one(self):
        s = scaling.ExponentialScale(0.0, 1.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.0507566928863525
        for actual, expected in zip(newfitness, [base**x - 1 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_six(self):
        s = scaling.ExponentialScale(5.0, 6.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.0507566928863525
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine(self):
        s = scaling.ExponentialScale(5.0, 9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_int(self):
        s = scaling.ExponentialScale(5, 9)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_five_nine_int_cuda(self):
        s = scaling.ExponentialScale(5, 9)
        f = t.arange(15, dtype=t.float64, device='cuda')
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-5)

    def test_five_nine_callback(self):
        s = scaling.ExponentialScale(lambda *_,**__: 5.0, lambda *_,**__: 9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_callback_int(self):
        s = scaling.ExponentialScale(lambda *_,**__: 5, lambda *_,**__: 9)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x+4 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_negative(self):
        s = scaling.ExponentialScale(-13.0, -9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.1218284368515015
        for actual, expected in zip(newfitness, [base**x-14 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)


if __name__ == '__main__':
    unittest.main()
