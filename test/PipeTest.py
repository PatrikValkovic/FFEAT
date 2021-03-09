###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

class PipeTest(unittest.TestCase):
    def test_one_argument(self):
        p = ffeat.Pipe()
        result, kargs = p(4)
        self.assertSequenceEqual(result, [4])

    def test_more_arguments(self):
        p = ffeat.Pipe()
        result, kargs = p(4,8,13)
        self.assertSequenceEqual(result, [4,8,13])

    def test_no_arguments(self):
        p = ffeat.Pipe()
        result, kargs = p()
        self.assertEqual(len(result), 0)

    def test_one_argument_and_kargs(self):
        p = ffeat.Pipe()
        args, kargs = p(13, something=True)
        self.assertSequenceEqual(args, [13])
        self.assertDictEqual(kargs, {"something": True})

    def test_more_arguments_and_kargs(self):
        p = ffeat.Pipe()
        args, kargs = p(4,8,13, something=True, karg="hello")
        self.assertSequenceEqual(args, [4,8,13])
        self.assertDictEqual(kargs, {"something": True, "karg": "hello"})


if __name__ == '__main__':
    unittest.main()
