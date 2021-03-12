###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
import unittest
import ffeat
import torch as t


class SimpleTest(unittest.TestCase):
    @unittest.skip("Irelevant")
    def test_handcomposed(self):
        alg = ffeat.flow.Sequence(
            ffeat.strategies.initialization.Uniform(100, 5.0, -5.0, 40, device='cpu'),
            ffeat.flow.Repeat(
                ffeat.flow.Sequence(
                    ffeat.strategies.evaluation.Evaluation(lambda x: t.sum(t.pow(x, 2), dim=-1)),
                    ffeat.strategies.selection.Tournament(1.0),
                    ffeat.strategies.mutation.AddFromNormal(0.001, 0.1),
                    ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True)
                ),
                max_iterations=100,
                loop_arguments=True
            )
        )
        alg()

    @unittest.skip("Irelevant")
    def test_in_Strategy(self):
        alg = ffeat.strategies.Strategy(
            ffeat.strategies.initialization.Uniform(100, 5.0, -5.0, 40, device='cuda'),
            ffeat.strategies.evaluation.Evaluation(lambda x: t.sum(t.pow(x, 2), dim=-1)),
            ffeat.strategies.selection.Tournament(1.0),
            ffeat.strategies.mutation.AddFromNormal(0.001, 0.1),
            ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True)
        )
        alg()


if __name__ == '__main__':
    unittest.main()
