###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest


import unittest
import torch as t
from ffeat import pso


class GridTest(unittest.TestCase):
    def test_one_linear_4D_can_call(self):
        c = pso.neighborhood.Grid('linear', 1, (17,19,23,29))
        n = c(t.rand((17*19*23*29,7)), t.rand((17*19*23*29,7)))

    def test_three_linear_4D_can_call(self):
        c = pso.neighborhood.Grid('linear', 3, (17,19,23,29))
        n = c(t.rand((17*19*23*29,7)), t.rand((17*19*23*29,7)))

    def test_seven_linear_4D_can_call(self):
        c = pso.neighborhood.Grid('linear', 7, (17,19,23,29))
        n = c(t.rand((17*19*23*29,7)), t.rand((17*19*23*29,7)))

    def test_seven_linear_4D_fraction_size(self):
        c = pso.neighborhood.Grid('linear', 0.0001, (17,19,23,29))
        n = c(t.rand((17*19*23*29,7)), t.rand((17*19*23*29,7)))


if __name__ == '__main__':
    unittest.main()
