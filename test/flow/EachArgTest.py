###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat


class EachArgTest(unittest.TestCase):
    def test_oneparam(self):
        p = ffeat.flow.EachArg(lambda x: x + 1)
        result, kargs = p(8)
        self.assertSequenceEqual(result, [9])

    def test_moreparams(self):
        p = ffeat.flow.EachArg(lambda x: x + 1)
        result, kargs = p(8, 12, 19, 20)
        self.assertSequenceEqual(result, [9, 13, 20, 21])


if __name__ == '__main__':
    unittest.main()
