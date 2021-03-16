###############################
#
# Created by Patrik Valkovic
# 3/15/2021
#
###############################
import unittest
import ffeat.utils.decay as decay


class PolynomialTest(unittest.TestCase):
    def test_power_1(self):
        d = decay.Polynomial(1.0, 0.1, 1.0)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 0.9*(1-x/10)+0.1, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_power_small(self):
        d = decay.Polynomial(1.0, 0.1, 0.01)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 0.9*(1-x/10)**0.01+0.1, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_power_big(self):
        d = decay.Polynomial(1.0, 0.1, 13.2)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 0.9*(1-x/10)**13.2+0.1, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_higher_power_1(self):
        d = decay.Polynomial(9.7, 5.6, 1.0)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 4.1*(1-x/10)+5.6, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_higher_power_small(self):
        d = decay.Polynomial(9.7, 5.6, 0.14)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]
        for expected, actual in zip(map(lambda x: 4.1*(1-x/10)**0.14+5.6, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_higher_power_big(self):
        d = decay.Polynomial(9.7, 5.6, 13.2)
        result = [d(iteration=x, max_iteration=10) for x in range(10)]

        for expected, actual in zip(map(lambda x: 4.1*(1-x/10)**13.2+5.6, range(10)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_more_steps(self):
        d = decay.Polynomial(93.7, 52.6, 0.85)
        result = [d(iteration=x, max_iteration=100) for x in range(100)]
        for expected, actual in zip(map(lambda x: 41.1*(1-x/100)**0.85+52.6, range(100)), result):
            self.assertLess(abs(expected - actual), 1e-6)

    def test_int(self):
        d = decay.Polynomial(93.7, 52.6, 0.85, result_type=int)
        result = [d(iteration=x, max_iteration=100) for x in range(100)]
        for expected, actual in zip(map(lambda x: int(41.1*(1-x/100)**0.85+52.6), range(100)), result):
            self.assertEqual(expected, actual)

    def test_no_max_iter(self):
        d = decay.Polynomial(9.7, 5.6, 0.014)
        with self.assertRaises(ValueError):
            d(iteration=12, max_iteration=None)


if __name__ == '__main__':
    unittest.main()
