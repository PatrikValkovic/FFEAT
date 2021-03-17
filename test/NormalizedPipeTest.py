###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

class NormalizedPipeTest(unittest.TestCase):
    def test_valid(self):
        n = ffeat.NormalizedPipe()
        param = (1,3,8), {"something": True}
        result = n(param)
        self.assertEqual(result, param)

    def test_onlyargs(self):
        n = ffeat.NormalizedPipe()
        result = n((1,3,8))
        self.assertEqual(result, ((1,3,8), {}))

    def test_only_one_arg(self):
        n = ffeat.NormalizedPipe()
        result = n(13)
        self.assertEqual(result, ((13,), {}))

    def test_none(self):
        n = ffeat.NormalizedPipe()
        result = n(None)
        self.assertEqual(result, (tuple(), {}))



if __name__ == '__main__':
    unittest.main()
