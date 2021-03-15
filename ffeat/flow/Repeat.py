###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Any, Dict
from ffeat import Pipe

class Repeat(Pipe):
    def __init__(self, pipe: Pipe,
                 max_iterations=None,
                 *,
                 loop_arguments=True,
                 identifier=None):
        self._pipe = pipe
        self._max_iterations = max_iterations
        self._loop_arguments = loop_arguments
        self._identifier = identifier

    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        def _iter():
            if self._max_iterations is None:
                while True:
                    yield None
            yield from range(self._max_iterations)

        break_identifier = object()
        def _break():
            raise StopIteration(break_identifier)
        key_name = "break" if self._identifier is None else f"{self._identifier}_break"
        previous_break = kwargs.get('break', None)
        previous_iter = kwargs.get('iteration', None)
        previous_max_iters = kwargs.get('max_iteration', None)
        kwargs.update({"break": _break, key_name: _break, 'max_iteration': self._max_iterations})

        cargs = list(args)
        ckargs = dict(kwargs)

        try:
            current_iter = 0
            for _ in _iter():
                current_iter += 1
                ckargs['iteration'] = current_iter
                cargs, ckargs = self._pipe(
                    *(args if not self._loop_arguments else cargs),
                    **(kwargs if not self._loop_arguments else ckargs)
                )
        except StopIteration as e:
            if e.args[0] != break_identifier:
                raise e

        if 'break' in ckargs:
            del ckargs['break']
        if key_name in ckargs:
            del ckargs[key_name]
        if previous_break is not None:
            ckargs['break'] = previous_break
        if previous_iter is not None:
            ckargs['iteration'] = previous_iter
        ckargs['max_iteration'] = previous_max_iters
        return cargs, ckargs
