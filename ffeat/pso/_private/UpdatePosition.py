###############################
#
# Created by Patrik Valkovic
# 3/19/2021
#
###############################
from ffeat import Pipe, STANDARD_REPRESENTATION


class UpdatePosition(Pipe):
    """
    Update particles position based on the velocity.
    """

    def __call__(self, position, velocity, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Updated particles positions based on the velocity.
        :param position: Current positions of the particles.
        :param velocity: Particles velocities.
        :param kwargs: Keyword arguments.
        :return: Updated positions.
        """
        position.add_(velocity)
        return (position,), kwargs
