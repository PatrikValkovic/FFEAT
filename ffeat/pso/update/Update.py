###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from ffeat import Pipe, STANDARD_REPRESENTATION


class Update(Pipe):
    """
    Base class for PSO update algorithms.
    """
    def __call__(self, position, velocities,
                 fitness_gbest, positions_gbest,
                 fitness_lbest, positions_lbest, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Method that should ve overwritten by derived class.
        :param position: Particles positions.
        :param velocities: Particles velocities.
        :param fitness_gbest: Fitness of particles global best positions.
        :param positions_gbest: Particles global best positions.
        :param fitness_lbest: Fitness of particles local best positions.
        :param positions_lbest: Particles local best positions.
        :param kwargs: Keyword arguments.
        :return: None
        """
        raise NotImplementedError()
