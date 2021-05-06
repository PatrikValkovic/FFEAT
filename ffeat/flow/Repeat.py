###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from ffeat import Pipe, STANDARD_REPRESENTATION

class Repeat(Pipe):
    """
    Repeat pipe in a loop.
    Allows to specify, whether pipe's output should be passed back in the next iteration.
    Allows to break the loop prematurely.
    """
    def __init__(self, pipe: Pipe,
                 max_iterations=None,
                 *,
                 loop_arguments=True,
                 identifier=None):
        """
        Repeat pipe in a loop.
        Allows to specify, whether pipe's output should be passed back in the next iteration.
        Allows to break the loop prematurely.
        :param pipe: Pipe to execute in loop.
        :param max_iterations: Maximum number of iterations or None, if the loop should be infinite.
        :param loop_arguments: True if output of the pipe should be passed back to it in the next iteration.
        :param identifier: Loop identifier, allowing to break the loop using `{identifier}_break` keyword arguments.
        """
        self._pipe = pipe
        self._max_iterations = max_iterations
        self._loop_arguments = loop_arguments
        self._identifier = identifier

    def __call__(self, *args, **kwargs) -> STANDARD_REPRESENTATION:
        """
        Run the pipe in parallel.
        Add `iteration` and `max_iteration` keyword arguments to the pipe.
        Add `break` and `{identifier}_break` keyword arguments to the pipe. These are callable objects allowing early
        termination of the loop. When called, the input to the pipe in the current iteration is returned.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: Output from the last execution of the pipe.
        """
        def _iter():
            if self._max_iterations is None:
                while True:
                    yield None
            yield from range(self._max_iterations)

        break_identifier = object()
        def _break(*args, **kwargs):
            raise StopIteration(break_identifier, args, kwargs)
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
            if len(e.args[1]) > 0:
                cargs = e.args[1]
            ckargs.update(e.args[2])

        if 'break' in ckargs:
            del ckargs['break']
        if key_name in ckargs:
            del ckargs[key_name]
        if previous_break is not None:
            ckargs['break'] = previous_break
        ckargs['iteration'] = previous_iter
        ckargs['max_iteration'] = previous_max_iters
        return cargs, ckargs
