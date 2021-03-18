###############################
#
# Created by Patrik Valkovic
# 3/18/2021
#
###############################
from typing import Tuple, Any, Dict, Optional
from ffeat import Pipe


class Replace(Pipe):
    def __init__(self, pipe: Pipe, param_index: int, num_params: Optional[int] = 1):
        self._pipe = pipe
        self._param_index = param_index
        self._num_params = num_params

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        nparams, nkargs = self._pipe(*args, **kwargs)
        if self._num_params is not None and len(nparams) != self._num_params:
            raise ValueError(f"Returned {len(nparams)} parameters, but expected was {self._num_params}")

        follow_index = self._param_index+(self._num_params if self._num_params is not None else 1)
        to_return_params = [
            *args[:self._param_index],
            *nparams,
            *args[follow_index:]
        ]
        kwargs.update(nkargs)
        return tuple(to_return_params), kwargs
