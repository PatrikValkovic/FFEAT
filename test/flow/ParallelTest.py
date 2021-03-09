###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

def _f1(*args, **kargs):
    return tuple(map(lambda x: x + 1, args)), kargs

def _f2(increment):
    return lambda *args, **kwargs: (tuple(map(lambda x: x + increment, args)), kwargs)

class ParallelTest(unittest.TestCase):
    def test_one_oneparam(self):
        p = ffeat.flow.Parallel(_f1)
        result, kargs = p(5)
        self.assertSequenceEqual(result, [6])

    def test_one_moreparams(self):
        p = ffeat.flow.Parallel(_f1)
        result, kargs = p(5,9,15,36)
        self.assertSequenceEqual(result, [6,10,16,37])

    def test_more_oneparam(self):
        p = ffeat.flow.Parallel(_f2(1), _f2(11), _f2(101))
        result, kargs = p(5)
        self.assertSequenceEqual(result, [6, 16, 106])

    def test_more_moreparam(self):
        p = ffeat.flow.Parallel(_f2(1), _f2(11), _f2(101))
        result, kargs = p(3, 4, 7)
        self.assertSequenceEqual(result, [4, 5, 8, 14, 15, 18, 104, 105, 108])


if __name__ == '__main__':
    unittest.main()
