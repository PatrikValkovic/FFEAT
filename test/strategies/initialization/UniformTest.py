###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
import unittest
import torch as t
from ffeat.strategies import initialization


class UniformTest(unittest.TestCase):
    def test_population_size_match(self):
        i = initialization.Uniform(51, -2.0, 2.0, 312)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 312))

    def test_population_dimension_match(self):
        i = initialization.Uniform(51, -2.0, 2.0, (8,7))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 8, 7))

    def test_dimension_from_min(self):
        i = initialization.Uniform(51, t.full((312,), -2.0), 2.0)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 312))

    def test_dimension_from_min_multidimensional(self):
        i = initialization.Uniform(51, t.full((8,7), -2.0), 2.0)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 8, 7))

    def test_dimension_from_max(self):
        i = initialization.Uniform(51, -2.0, t.full((312,), 2.0))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 312))

    def test_dimension_from_max_multidimensional(self):
        i = initialization.Uniform(51, -2.0, t.full((8,7), 2.0))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 8, 7))

    def test_missing_dimension(self):
        with self.assertRaises(ValueError):
            initialization.Uniform(51, -2.0, 2.0)

    def test_max_min_dimensions_not_match(self):
        with self.assertRaises(ValueError):
            initialization.Uniform(51, t.full((4,5), -2.0), t.full((8,7), 2.0))

    def test_min_dimension_not_match(self):
        with self.assertRaises(ValueError):
            initialization.Uniform(51, t.full((4,5), -2.0), 2.0, (4,6))

    def test_max_dimension_not_match(self):
        with self.assertRaises(ValueError):
            initialization.Uniform(51, -2.0, t.full((4,5), 2.0), (4,6))

    def test_max_smaller_than_min(self):
        with self.assertRaises(ValueError):
            initialization.Uniform(51, 2.0, -2.0, (4,6))

    def test_min_int(self):
        i = initialization.Uniform(51, -2, 2.0, (4,6))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_min_float(self):
        i = initialization.Uniform(51, -2, 2.0, (4,6))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_min_list(self):
        i = initialization.Uniform(51, [[-2.0]*6]*4, 2.0)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_min_tensor(self):
        i = initialization.Uniform(51, t.full((4,6), -2.0), 2.0)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_max_int(self):
        i = initialization.Uniform(51, -2.0, 2, (4,6))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_max_float(self):
        i = initialization.Uniform(51, -2.0, 2.0, (4,6))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_max_list(self):
        i = initialization.Uniform(51, -2.0, [[2.0]*6]*4)
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_max_tensor(self):
        i = initialization.Uniform(51, -2.0, t.full((4,6), 2.0))
        pop, kargs = i()
        self.assertEqual(pop[0].shape, (51, 4, 6))

    def test_max_works(self):
        i = initialization.Uniform(51, -2.0, t.full((4,6), 2.0))
        for _ in range(100):
            pop, kargs = i()
            self.assertTrue(t.all(pop[0] < 2.0))

    def test_min_works(self):
        i = initialization.Uniform(51, t.full((4,6), -2.0), 2.0)
        for _ in range(100):
            pop, kargs = i()
            self.assertTrue(t.all(pop[0] >= -2.0))

    def test_shifted(self):
        i = initialization.Uniform(51, 3.0, t.full((4,6), 5.0))
        for _ in range(100):
            pop, kargs = i()
            self.assertTrue(t.all(pop[0] >= 3.0))
            self.assertTrue(t.all(pop[0] < 5.0))

    def test_float16_type(self):
        i = initialization.Uniform(51, 3.0, t.full((4,6), 5.0), dtype=t.float16)
        pop, kargs = i()
        self.assertEqual(pop[0].dtype, t.float16)

    def test_long_type(self):
        i = initialization.Uniform(51, 3.0, t.full((4,6), 5.0), dtype=t.long)
        pop, kargs = i()
        self.assertEqual(pop[0].dtype, t.long)

    def test_int8_type(self):
        i = initialization.Uniform(51, 3.0, t.full((4,6), 5.0), dtype=t.int8)
        pop, kargs = i()
        self.assertEqual(pop[0].dtype, t.int8)

    @unittest.skipIf(not t.cuda.is_available(), "CUDA not available")
    def test_on_cuda(self):
        i = initialization.Uniform(51, 3.0, 5.0, (7,8), device='cuda')
        pop, kargs = i()
        self.assertEqual(pop[0].device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), "CUDA not available")
    def test_device_from_min(self):
        i = initialization.Uniform(51, t.full((7,8), 3.0, device='cuda'), 5.0)
        pop, kargs = i()
        self.assertEqual(pop[0].device, t.device('cuda:0'))

    @unittest.skipIf(not t.cuda.is_available(), "CUDA not available")
    def test_device_from_max(self):
        i = initialization.Uniform(51, 3.0, t.full((7,8), 5.0, device='cuda'))
        pop, kargs = i()
        self.assertEqual(pop[0].device, t.device('cuda:0'))




if __name__ == '__main__':
    unittest.main()
