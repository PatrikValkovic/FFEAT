###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import unittest
import ffeat.decay as decay


class LinearTest(unittest.TestCase):
    def test_start_end(self):
        d = decay.Linear(1.0, 0.0)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_start_step(self):
        d = decay.Linear(1.0, step=0.1)
        result = [d(iteration=x) for x in range(10)]
        for expected, actual in zip([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_start_int(self):
        d = decay.Linear(10.0, 5.0, result_type=int)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        self.assertSequenceEqual(result, [10, 9, 9, 8, 8, 7, 7, 6, 6, 5])

    def test_no_step_provided(self):
        with self.assertRaises(ValueError):
            decay.Linear(1.0)


if __name__ == '__main__':
    unittest.main()
