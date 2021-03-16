###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
import unittest
import math
import torch as t
from ffeat.utils import scaling


class ExponentialScaleTest(unittest.TestCase):
    def test_zero_one(self):
        s = scaling.LogScale(0.0, 1.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 15
        for actual, expected in zip(newfitness, [math.log(x+1, base) for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_six(self):
        s = scaling.LogScale(5.0, 6.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 15
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine(self):
        s = scaling.LogScale(5.0, 9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_int(self):
        s = scaling.LogScale(5, 9)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_five_nine_int_cuda(self):
        s = scaling.LogScale(5, 9)
        f = t.arange(15, dtype=t.float64, device='cuda')
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_callback(self):
        s = scaling.LogScale(lambda *_,**__: 5.0, lambda *_,**__: 9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_callback_int(self):
        s = scaling.LogScale(lambda *_,**__: 5, lambda *_,**__: 9)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)+5.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_negative(self):
        s = scaling.LogScale(-13.0, -9.0)
        f = t.arange(15, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        base = 1.9679896831512451
        for actual, expected in zip(newfitness, [math.log(x+1, base)-13.0 for x in range(15)]):
            self.assertLess(abs(actual - expected), 1e-6)


if __name__ == '__main__':
    unittest.main()
