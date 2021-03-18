###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Any, Dict
import math
import torch as t
from ffeat import Pipe, flow
from .neighborhood.Neighborhood import Neighborhood
from .update.Update import Update
from ._private.UpdateLocalBest import UpdateLocalBest
from ._private.UpdateGlobalBest import UpdateGlobalBest


class PSO(Pipe):
    def __init__(self,
                 position_initialization: Pipe,
                 velocity_initialization: Pipe,
                 evaluation: Pipe,
                 neighborhood_definition: Neighborhood,
                 velocity_update: Update,
                 iterations: int = 100):
        self.__flow = flow.Sequence(
            flow.Parallel(
                position_initialization,
                velocity_initialization,
            ),
            # position, velocities
            flow.Concat(
                lambda pop, *_, **__: ((t.full((len(pop),), math.inf, device=pop.device),), {}),
                lambda pop, *_, **__: ((t.full(pop.shape[1:], 0, device=pop.device, dtype=pop.dtype),), {}),
                lambda pop, *_, **__: ((t.full((len(pop),), math.inf, device=pop.device),), {}),
                lambda pop, *_, **__: ((t.full(pop.shape[1:], 0, device=pop.device, dtype=pop.dtype),), {}),
            ),
            # position, velocities, fitness_gbest, positions_gbest, fintess_lbest, positions_lbest
            flow.Repeat(
                flow.Sequence(
                    # position, velocities, fitness_gbest, positions_gbest, fintess_lbest, positions_lbest
                    evaluation,
                    # fitness, position, velocities, fitness_gbest, positions_gbest, fitness_lbest, positions_lbest
                    flow.Replace(
                        flow.Sequence(
                            flow.Select(0, 1, 5, 6),
                            # fitness, position, fitness_lbest, positions_lbest
                            UpdateLocalBest(),
                        ),
                        param_index=5,
                        num_params=2,
                    ),
                    # fitness, position, velocities, fitness_gbest, positions_gbest, fitness_lbest, positions_lbest
                    flow.Replace(
                        flow.Sequence(
                            flow.Select(0, 1, 3, 4),
                            # fitness, position, fitness_gbest, positions_gbest
                            UpdateGlobalBest(neighborhood_definition),
                        ),
                        param_index=3,
                        num_params=2,
                    ),
                    flow.Select(list(range(1, 7))),
                    # position, velocities, fitness_gbest, positions_gbest, fitness_lbest, positions_lbest
                    flow.Replace(
                        velocity_update,
                        param_index=0,
                        num_params=2,
                    )
                ),
                max_iterations=iterations,
                loop_arguments=True,
                identifier='ffeat'
            )
        )

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return self.__flow(*args, **kwargs)
