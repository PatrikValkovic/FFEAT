###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat

class FlowSelectTests(unittest.TestCase):
    def test_one(self):
        s = ffeat.flow.Select(2)
        result, kargs = s(8,1,6,2,4)
        self.assertSequenceEqual(result, [6])

    def test_one_in_list(self):
        s = ffeat.flow.Select([2])
        result, kargs = s(8,1,6,2,4)
        self.assertSequenceEqual(result, [6])

    def test_more(self):
        s = ffeat.flow.Select(2, 4, 0)
        result, kargs = s(8,1,6,2,4)
        self.assertSequenceEqual(result, [6, 4, 8])

    def test_more_in_list(self):
        s = ffeat.flow.Select([2, 4, 0])
        result, kargs = s(8,1,6,2,4)
        self.assertSequenceEqual(result, [6, 4, 8])

    def test_one_out(self):
        s = ffeat.flow.Select(8)
        with self.assertRaises(ValueError):
            s(8,1,6,2,4)

    def test_one_out_in_list(self):
        s = ffeat.flow.Select([8])
        with self.assertRaises(ValueError):
            s(8,1,6,2,4)

    def test_more_out(self):
        s = ffeat.flow.Select(2, 4, 8, 0)
        with self.assertRaises(ValueError):
            s(8,1,6,2,4)

    def test_more_out_in_list(self):
        s = ffeat.flow.Select([8, 2, 4, 0])
        with self.assertRaises(ValueError):
            s(8,1,6,2,4)


    def test_one_kargs(self):
        s = ffeat.flow.Select(2)
        result, kargs = s(8,1,6,2,4, something=True, else2=18)
        self.assertSequenceEqual(result, [6])
        self.assertDictEqual(kargs, {"something": True, "else2": 18})

    def test_one_in_list_kargs(self):
        s = ffeat.flow.Select([2])
        result, kargs = s(8,1,6,2,4, something=True, else2=18)
        self.assertSequenceEqual(result, [6])
        self.assertDictEqual(kargs, {"something": True, "else2": 18})

    def test_more_kargs(self):
        s = ffeat.flow.Select(2, 4, 0)
        result, kargs = s(8,1,6,2,4, something=True, else2=18)
        self.assertSequenceEqual(result, [6, 4, 8])
        self.assertDictEqual(kargs, {"something": True, "else2": 18})

    def test_more_in_list_kargs(self):
        s = ffeat.flow.Select([2, 4, 0])
        result, kargs = s(8,1,6,2,4, something=True, else2=18)
        self.assertSequenceEqual(result, [6, 4, 8])
        self.assertDictEqual(kargs, {"something": True, "else2": 18})


if __name__ == '__main__':
    unittest.main()
