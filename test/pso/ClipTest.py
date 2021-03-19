###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
import unittest
import torch as t
from ffeat import pso


class ClipTest(unittest.TestCase):
    def test_position_by_int(self):
        c = pso.clip.Position(-3, 3)
        pop = t.randn((1000,400)) * 5
        bigger = pop > 3
        lower = pop < -3
        (newpop, newfitness), kargs = c(pop, t.rand((1000,400)))
        self.assertTrue(t.all(newpop <= 3))
        self.assertTrue(t.all(newpop >= -3))
        self.assertTrue(t.all(newpop[bigger] == 3))
        self.assertTrue(t.all(newpop[lower] == -3))
        self.assertIs(newpop, pop)

    def test_position_by_float(self):
        c = pso.clip.Position(-2.6, 2.6)
        pop = t.randn((1000,400)) * 5
        bigger = pop > 2.6
        lower = pop < -2.6
        (newpop, newfitness), kargs = c(pop, t.rand((1000,400)))
        self.assertTrue(t.all(newpop <= 2.6))
        self.assertTrue(t.all(newpop >= -2.6))
        self.assertTrue(t.all(newpop[bigger] == 2.6))
        self.assertTrue(t.all(newpop[lower] == -2.6))
        self.assertIs(newpop, pop)

    def test_velocity_by_int_value(self):
        c = pso.clip.VelocityValue(-3, 3)
        velocities = t.randn((1000,400)) * 5
        bigger = velocities > 3
        lower = velocities < -3
        (new_velocity,), kargs = c(velocities)
        self.assertTrue(t.all(new_velocity <= 3))
        self.assertTrue(t.all(new_velocity >= -3))
        self.assertTrue(t.all(new_velocity[bigger] == 3))
        self.assertTrue(t.all(new_velocity[lower] == -3))
        self.assertIs(velocities, new_velocity)

    def test_velocity_by_float_value(self):
        c = pso.clip.VelocityValue(-2.6, 3.1)
        velocities = t.randn((1000,400)) * 5
        bigger = velocities > 3.1
        lower = velocities < -2.6
        (new_velocity,), kargs = c(velocities)
        self.assertTrue(t.all(new_velocity <= 3.1))
        self.assertTrue(t.all(new_velocity >= -2.6))
        self.assertTrue(t.all(new_velocity[bigger] == 3.1))
        self.assertTrue(t.all(new_velocity[lower] == -2.6))
        self.assertIs(velocities, new_velocity)

    def test_velocity_by_float_norm(self):
        c = pso.clip.VelocityNorm(2.3)
        velocities = t.randn((1000,400)) * 5
        (new_velocity,), kargs = c(velocities)
        self.assertTrue(t.all(t.norm(new_velocity, dim=1) <= 2.3+1e-3))
        self.assertIs(velocities, new_velocity)

    def test_velocity_by_float_int(self):
        c = pso.clip.VelocityNorm(3)
        velocities = t.randn((1000,400)) * 5
        (new_velocity,), kargs = c(velocities)
        self.assertTrue(t.all(t.norm(new_velocity, dim=1) <= 3.0+1e-3))
        self.assertIs(velocities, new_velocity)



if __name__ == '__main__':
    unittest.main()
