###############################
#
# Created by Patrik Valkovic
# 3/20/2021
#
###############################
import unittest
from ffeat.pso.update.Update import Update
from ffeat.pso.neighborhood.Neighborhood import Neighborhood


class ExceptionsTest(unittest.TestCase):
    def test_base_update_not_implemented(self):
        u = Update()
        with self.assertRaises(NotImplementedError):
            u(position=None, velocities=None,
              fitness_gbest=None, positions_gbest=None,
              fitness_lbest=None, positions_lbest=None)

    def test_base_neighborhood_not_implemented(self):
        n = Neighborhood()
        with self.assertRaises(NotImplementedError):
            n(fitnesses=None, position=None)


if __name__ == '__main__':
    unittest.main()
