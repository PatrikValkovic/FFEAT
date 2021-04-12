###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest
import torch as t
from ffeat import pso


class Grid2DTest(unittest.TestCase):
    def test_one_linear(self):
        #  0  1  2  3
        #  4  5  6  7
        #  8  9 10 11
        # 12 13 14 15
        # 16 17 18 19
        c = pso.neighborhood.Grid2D('linear', 1, (4,5))
        n = c(t.rand((20,2)), t.rand((20,2)))
        expected = [
            [16,  3,  1,  4],
            [17,  0,  2,  5],
            [18,  1,  3,  6],
            [19,  2,  0,  7],
            [ 0,  7,  5,  8],
            [ 1,  4,  6,  9],
            [ 2,  5,  7, 10],
            [ 3,  6,  4, 11],
            [ 4, 11,  9, 12],
            [ 5,  8, 10, 13],
            [ 6,  9, 11, 14],
            [ 7, 10,  8, 15],
            [ 8, 15, 13, 16],
            [ 9, 12, 14, 17],
            [10, 13, 15, 18],
            [11, 14, 12, 19],
            [12, 19, 17,  0],
            [13, 16, 18,  1],
            [14, 17, 19,  2],
            [15, 18, 16,  3]
        ]
        for actual, expected in zip(n.tolist(), expected):
            self.assertCountEqual(actual, expected)

    def test_three_linear_can_call(self):
        c = pso.neighborhood.Grid2D('linear', 3, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))

    def test_three_diamond_can_call(self):
        c = pso.neighborhood.Grid2D('diamond', 1, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))

    def test_five_diamond_can_call(self):
        c = pso.neighborhood.Grid2D('diamond', 5, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))

    def test_three_compact_can_call(self):
        c = pso.neighborhood.Grid2D('compact', 3, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))

    def test_five_compact_can_call(self):
        c = pso.neighborhood.Grid2D('compact', 5, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))

    def test_five_compact_fraction_size(self):
        c = pso.neighborhood.Grid2D('compact', 0.01, (31, 33))
        c(t.rand((31*33, 2)), t.rand((31*33, 2)))


if __name__ == '__main__':
    unittest.main()
