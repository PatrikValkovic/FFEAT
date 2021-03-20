###############################
#
# Created by Patrik Valkovic
# 3/20/2021
#
###############################
import unittest
import torch as t
from ffeat.pso import neighborhood


class InvalidSizeTest(unittest.TestCase):
    def test_2D_size_not_match(self):
        g = neighborhood.Grid2D('linear', 2, (11,13))
        p, f = t.randn((11*13+1, 40)), t.rand((11*13+1, 40))
        with self.assertRaises(ValueError):
            g(f, p)

    def test_2D_invalid_mode(self):
        g = neighborhood.Grid2D('something', 2, (11,13))
        p, f = t.randn((11*13, 40)), t.rand((11*13, 40))
        with self.assertRaises(ValueError):
            g(f, p)

    def test_2D_more_dims(self):
        with self.assertRaises(ValueError):
            neighborhood.Grid2D('linear', 2, (11,13,17))

    def test_4D_size_not_match(self):
        g = neighborhood.Grid('linear', 2, (11,13,17,21))
        p, f = t.randn((11*13*17+1,7)), t.rand((11*13*17*21+1,7))
        with self.assertRaises(ValueError):
            g(f, p)

    def test_grid_compact_not_implemented(self):
        g = neighborhood.Grid('compact', 2, (11,13,17,21))
        p, f = t.randn((11*13*17,7)), t.rand((11*13*17*21,7))
        with self.assertRaises(NotImplementedError):
            g(f, p)

    def test_grid_diamond_not_implemented(self):
        g = neighborhood.Grid('diamond', 2, (11,13,17,21))
        p, f = t.randn((11*13*17,7)), t.rand((11*13*17*21,7))
        with self.assertRaises(NotImplementedError):
            g(f, p)


if __name__ == '__main__':
    unittest.main()
