###############################
#
# Created by Patrik Valkovic
# 3/16/2021
#
###############################
from typing import Tuple, Any, Dict, Union, Callable, List
import torch as t
from ffeat import Pipe

_IFU = Union[int, float]


class Replace(Pipe):
    def __init__(self,
                 distribution: t.distributions.Distribution,
                 mutation_rate: Union[float, Callable[..., float]],
                 in_place: bool = True
                 ):
        if isinstance(mutation_rate, float) and (mutation_rate < 0.0 or mutation_rate > 1.0):
            raise ValueError("Mutation rate must be in the range [0.0, 1.0]")
        self._mutation_rate = self._handle_parameter(mutation_rate)
        self._distribution = distribution
        self.in_place = in_place

    def __call__(self, population, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        dist = self._distribution
        mutation_rate = self._mutation_rate(population, *args, **kwargs)
        if mutation_rate < 0.0 or mutation_rate > 1.0:
            raise ValueError("Mutation rate must be in the range [0.0, 1.0]")
        dist_shape = dist.sample((0,)).shape[1:]
        should_modify = t.rand(population.shape, device=population.device) < mutation_rate
        new_values = dist.sample(sample_shape=population.shape[:len(population.shape)-len(dist_shape)]).type(population.dtype).to(population.device)
        if new_values.shape != population.shape:
            raise ValueError(f"Invalid distribution shape, expected {population.shape} but received {new_values.shape}")
        if not self.in_place:
            population = t.clone(population)
        population[should_modify] = new_values[should_modify]
        return (population, *args), kwargs


class ReplaceUniform(Replace):
    def __init__(self,
                 min: Union[_IFU, List[_IFU], t.Tensor],
                 max: Union[_IFU, List[_IFU], t.Tensor],
                 mutation_rate: Union[float, Callable[..., float]],
                 in_place: bool = True):
        tmp_min, tmp_max = min, max
        if isinstance(min, list) or isinstance(min, float) or isinstance(min, int):
            tmp_min = t.tensor(min)
        if isinstance(max, list) or isinstance(max, float) or isinstance(max, int):
            tmp_max = t.tensor(max)
        tmp_min = tmp_min.type(t.float32)
        tmp_max = tmp_max.type(t.float32)
        if isinstance(min, t.Tensor):
            tmp_max = tmp_max.type(min.dtype).to(min.device)
        if isinstance(max, t.Tensor):
            tmp_min = tmp_min.type(max.dtype).to(max.device)
        super().__init__(t.distributions.Uniform(tmp_min, tmp_max), mutation_rate, in_place)
