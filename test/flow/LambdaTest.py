###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

class LambdaTest(unittest.TestCase):
    def test_oneadd(self):
        l = ffeat.flow.Lambda(lambda x: x + 1)
        result, kargs = l(5)
        self.assertSequenceEqual(result, [6])

    def test_alladd(self):
        l = ffeat.flow.Lambda(lambda *args: tuple(map(lambda x: x + 1, args)))
        result, kargs = l(5,8,85)
        self.assertSequenceEqual(result, [6,9,86])

    def test_dictlamb(self):
        def _l(*args, **kwargs):
            return tuple(args), {"lambda": 1, **kwargs}
        l = ffeat.flow.Lambda(_l)
        args, kargs = l(7,9,2, something=True)
        self.assertSequenceEqual(args, [7, 9, 2])
        self.assertDictEqual(kargs, {"something": True, "lambda": 1})

    def test_no_return_value(self):
        def _l(*args, **kwargs):
            pass
        l = ffeat.flow.Lambda(_l)
        args, kargs = l(7, 9, 2, something=True)
        self.assertEqual(args, tuple())
        self.assertDictEqual(kargs, {})


if __name__ == '__main__':
    unittest.main()
