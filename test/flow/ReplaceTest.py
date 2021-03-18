###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
import unittest
from ffeat.flow import Replace


class ReplaceTest(unittest.TestCase):
    def test_one_param(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,), {}
        r = Replace(_fn, 4)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,20,5,6,7,8,9])

    def test_multiple(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 4, 3)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,20,30,40,7,8,9])

    def test_multiple_begin(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 0, 3)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [20,30,40,3,4,5,6,7,8,9])

    def test_multiple_end(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 9, 3)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,4,5,6,7,8,20,30,40])

    def test_multiple_append(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 10, 3)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,4,5,6,7,8,9,20,30,40])

    def test_one_no_length(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,), {}
        r = Replace(_fn, 4, None)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,20,5,6,7,8,9])

    def test_more_no_length(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 4, None)
        result, kargs = r(0,1,2,3,4,5,6,7,8,9)
        self.assertSequenceEqual(result, [0,1,2,3,20,30,40,5,6,7,8,9])

    def test_test_not_match(self):
        def _fn(*a, **k):
            self.assertEqual(len(a), 10)
            return (20,30,40), {}
        r = Replace(_fn, 4, 2)
        with self.assertRaises(ValueError):
            r(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


if __name__ == '__main__':
    unittest.main()
