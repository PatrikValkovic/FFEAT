###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
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

    def __call__(self, *args, **kwargs):
        def _iter():
            if self._max_iterations is None:
                while True:
                    yield None
            yield from range(self._max_iterations)

        break_identifier = object()
        def _break():
            raise StopIteration(break_identifier)
        key_name = "break" if self._identifier is None else f"{self._identifier}_break"
        kwargs.update({key_name: _break})
        kwargs.update({"break": _break})

        cargs = list(args)
        ckargs = dict(kwargs)

        try:
            for _ in _iter():
                cargs, ckargs = self._pipe(
                    *(args if not self._loop_arguments else cargs),
                    **(kwargs if not self._loop_arguments else ckargs)
                )
        except StopIteration as e:
            if e.args[0] != break_identifier:
                raise e

        return cargs, ckargs
