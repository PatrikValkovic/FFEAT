###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest
import torch as t
from ffeat import pso


class CircleTest(unittest.TestCase):
    def test_one(self):
        c = pso.neighborhood.Circle(1)
        n = c(t.rand((13,2)), t.rand((13,2)))
        expected = [
            [12,1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [10, 12],
            [11, 0],
        ]
        for actual, expected in zip(n.tolist(), expected):
            self.assertCountEqual(actual, expected)

    def test_three(self):
        c = pso.neighborhood.Circle(3)
        n = c(t.rand((13,2)), t.rand((13,2)))
        expected = [
            [10, 11, 12,  1,  2,  3],
            [11, 12,  0,  2,  3,  4],
            [12,  0,  1,  3,  4,  5],
            [ 0,  1,  2,  4,  5,  6],
            [ 1,  2,  3,  5,  6,  7],
            [ 2,  3,  4,  6,  7,  8],
            [ 3,  4,  5,  7,  8,  9],
            [ 4,  5,  6,  8,  9, 10],
            [ 5,  6,  7,  9, 10, 11],
            [ 6,  7,  8, 10, 11, 12],
            [ 7,  8,  9, 11, 12,  0],
            [ 8,  9, 10, 12,  0,  1],
            [ 9, 10, 11,  0,  1,  2],
        ]
        for actual, expected in zip(n.tolist(), expected):
            self.assertCountEqual(actual, expected)





if __name__ == '__main__':
    unittest.main()
