###############################
#
# Created by Patrik Valkovic
# 3/11/2021
#
###############################
import unittest
import ffeat


class SimpleTest(unittest.TestCase):
    def test_something(self):
        alg = ffeat.flow.Sequence(
            ffeat.strategies.initialization.Uniform(1000, 5.0, -5.0, 40, device='cuda'),
            ffeat.flow.Repeat(
                ffeat.flow.Sequence(
                    ffeat.strategies.mutation.AddFromNormal(0.001, 0.1),
                    ffeat.strategies.crossover.OnePoint1D(40, replace_parents=True)
                ),
                max_iterations=100,
                loop_arguments=True
            )
        )
        alg()


if __name__ == '__main__':
    unittest.main()
