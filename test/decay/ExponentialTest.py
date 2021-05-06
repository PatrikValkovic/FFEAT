###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import unittest
import ffeat.utils.decay as decay


class ExponentialTest(unittest.TestCase):
    def test_start_end(self):
        d = decay.Exponential(1.0, 0.1)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        rate = 0.7943282347242815
        for expected, actual in zip(map(lambda x: 1.0*rate**x, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_start_step(self):
        rate = 0.8
        d = decay.Exponential(1.0, rate=rate)
        result = [d(iteration=x) for x in range(10)]
        for expected, actual in zip(map(lambda x: 1.0*rate**x, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_start_bigger(self):
        d = decay.Exponential(10.0, 5.0)
        rate = 0.9330329915368074
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 10.0*rate**x, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_start_int(self):
        d = decay.Exponential(10.0, 5.0, result_type=int)
        rate = 0.9330329915368074
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: int(10.0*rate**x), range(10)), result):
            self.assertEqual(expected, actual)

    def test_no_step_provided(self):
        with self.assertRaises(ValueError):
            decay.Exponential(1.0)

    def test_no_max_steps(self):
        d = decay.Exponential(10.0, 2.0)
        with self.assertRaises(ValueError):
            d(iteration=5)


if __name__ == '__main__':
    unittest.main()
