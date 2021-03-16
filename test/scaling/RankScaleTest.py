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


class RankScaleTest(unittest.TestCase):
    def test_zero_one(self):
        s = scaling.RankScale(0.0, 1.0)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 100 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_six(self):
        s = scaling.RankScale(5.0, 6.0)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 100 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine(self):
        s = scaling.RankScale(5.0, 9.0)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_int(self):
        s = scaling.RankScale(5, 9)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    @unittest.skipIf(not t.cuda.is_available(), 'CUDA not available')
    def test_five_nine_int_cuda(self):
        s = scaling.RankScale(5, 9)
        f = t.arange(100, dtype=t.float32, device='cuda:0')
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_callback(self):
        s = scaling.RankScale(lambda *_,**__: 5.0, lambda *_,**__: 9.0)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_five_nine_callback_int(self):
        s = scaling.RankScale(lambda *_,**__: 5, lambda *_,**__: 9)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 + 5 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)

    def test_negative(self):
        s = scaling.RankScale(-13.0, -9.0)
        f = t.arange(100, dtype=t.float32)
        (newfitness,), kwargs = s(f)
        for actual, expected in zip(newfitness, [x / 25 - 13 for x in range(100)]):
            self.assertLess(abs(actual - expected), 1e-6)


if __name__ == '__main__':
    unittest.main()
