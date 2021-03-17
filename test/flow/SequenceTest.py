###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

class AddPipe(ffeat.NormalizedPipe):
    def __call__(self, argument, **kwargs):
        return super().__call__(argument+1)


class SequenceTest(unittest.TestCase):
    def test_oneadd(self):
        p = ffeat.flow.Sequence(AddPipe())
        result, kargs = p(5)
        self.assertSequenceEqual(result, [6])

    def test_moreadd(self):
        p = ffeat.flow.Sequence(
            AddPipe(), AddPipe(), AddPipe(), AddPipe()
        )
        result, kargs = p(5)
        self.assertSequenceEqual(result, [5+4])

    def test_moreargs(self):
        p = ffeat.flow.Sequence(AddPipe())
        with self.assertRaises(TypeError):
            p(5,8)


if __name__ == '__main__':
    unittest.main()
