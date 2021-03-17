###############################
#
# Created by Patrik Valkovic
# 3/17/2021
#
###############################
import unittest
import torch as t
import ffeat
from ffeat.utils.termination._Base import _BaseMetric


class StdBelowTest(unittest.TestCase):
    def test_no_improvement(self):
        executed = 0
        def _fn(_):
            nonlocal executed
            executed += 1
            return t.full((100,), abs(executed - 50), dtype=t.float32)
        ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5, 5, 40),
            ffeat.strategies.evaluation.Evaluation(_fn),
            ffeat.measure.FitnessMean(),
            ffeat.utils.termination.NoImprovement(
                ffeat.measure.FitnessMean.ARG_NAME,
                20
            ),
            iterations=100
        )()
        self.assertEqual(executed, 69)

    def test_metric_reached(self):
        executed = 0
        def _fn(_):
            nonlocal executed
            executed += 1
            return t.full((100,), abs(executed - 50), dtype=t.float32)
        ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, -5, 5, 40),
            ffeat.strategies.evaluation.Evaluation(_fn),
            ffeat.measure.FitnessMean(),
            ffeat.utils.termination.MetricReached(
                ffeat.measure.FitnessMean.ARG_NAME,
                20, 4
            ),
            iterations=100
        )()
        self.assertEqual(executed, 46)

    def test_std_bellow(self):
        executed = 0
        def _fn(_):
            nonlocal executed
            executed += 1
            return t.ones((1000,), dtype=t.float32) / executed
        ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(1000, -5, 5, 40),
            ffeat.strategies.evaluation.Evaluation(_fn),
            ffeat.measure.FitnessMean(),
            ffeat.utils.termination.StdBellow(
                ffeat.measure.FitnessMean.ARG_NAME,
                20, 0.01
            ),
            iterations=100
        )()
        self.assertEqual(executed, 36)

    def test_base_metric_not_implemented(self):
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(1000, -5, 5, 40),
            ffeat.strategies.evaluation.Evaluation(lambda x: t.sum(x ** 2, dim=-1)),
            ffeat.measure.FitnessMean(),
            _BaseMetric(
                ffeat.measure.FitnessMean.ARG_NAME,
                20
            ),
            iterations=100
        )
        with self.assertRaises(NotImplementedError):
            alg()

    def test_missing_measurement(self):
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(1000, -5, 5, 40),
            ffeat.strategies.evaluation.Evaluation(lambda x: t.sum(x ** 2, dim=-1)),
            ffeat.utils.termination.StdBellow(
                ffeat.measure.FitnessMean.ARG_NAME,
                20, 0.01
            ),
            iterations=100
        )
        with self.assertRaises(ValueError):
            alg()

    def test_not_used_within_algorithm(self):
        t = ffeat.utils.termination.StdBellow(
            ffeat.measure.FitnessMean.ARG_NAME,
            20, 0.01
        )
        d = {ffeat.measure.FitnessMean.ARG_NAME: 0}
        with self.assertRaises(ValueError):
            t(**d)


if __name__ == '__main__':
    unittest.main()
