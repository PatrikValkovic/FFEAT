###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
import unittest
import ffeat


class RepeatTest(unittest.TestCase):
    def test_finiteiterate_oneparam(self):
        count = 0
        def _f(arg, **kwargs):
            nonlocal count
            count += arg
            return (arg,), kwargs
        r = ffeat.flow.Repeat(_f, 57)
        r(8)
        self.assertEqual(count, 8*57)

    def test_finiteiterate_moreparam(self):
        count = 0
        def _f(arg1, arg2, arg3, **kwargs):
            nonlocal count
            count += arg1 + arg2 + arg3
            return (arg1, arg2, arg3), kwargs
        r = ffeat.flow.Repeat(_f, 182)
        r(8,7,6)
        self.assertEqual(count, 182*(8+7+6))

    def test_breakable(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 93:
                kwargs['break']()
            return args, kwargs
        r = ffeat.flow.Repeat(_f, 182)
        r(8,7,6,5)
        self.assertEqual(count, 93)

    def test_breakable_by_name(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 93:
                kwargs['loop1_break']()
            return args, kwargs
        r = ffeat.flow.Repeat(_f, 182, identifier="loop1")
        r(8,7,6,5)
        self.assertEqual(count, 93)

    def test_breakable_identified(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 93:
                kwargs['break']()
            return args, kwargs
        r = ffeat.flow.Repeat(_f, 182, identifier="loop1")
        r(8,7,6,5)
        self.assertEqual(count, 93)

    def test_breakable_infinite(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 241:
                return kwargs['break']()
            return args, kwargs

        r = ffeat.flow.Repeat(_f)
        r(8, 7, 6, 5)
        self.assertEqual(count, 241)

    def test_breakable_outer(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 241:
                return kwargs['outer_break']()
            return args, kwargs
        r = ffeat.flow.Repeat(ffeat.flow.Repeat(_f, 20), identifier='outer')
        r(8,7,6,5)
        self.assertEqual(count, 241)

    def test_loop_arguments(self):
        def _f(arg, **kwargs):
            return (arg+1,), kwargs
        r = ffeat.flow.Repeat(_f, max_iterations=241, loop_arguments=True)
        result, kargs = r(8)
        self.assertSequenceEqual(result, [241+8])

    def test_noloop_arguments(self):
        count = 0
        def _f(arg, **kwargs):
            nonlocal count
            count += 1
            return (arg+1,), kwargs
        r = ffeat.flow.Repeat(_f, max_iterations=241, loop_arguments=False)
        result, kargs = r(8)
        self.assertSequenceEqual(result, [8+1])
        self.assertEqual(count, 241)

    def test_breakable_with_return(self):
        count = 0
        def _f(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 241:
                return kwargs['break'](1,2,something1=True)
            return args, kwargs

        r = ffeat.flow.Repeat(_f)
        (one,two), kargs = r(8, 7, 6, 5)
        self.assertEqual(count, 241)
        self.assertEqual(one, 1)
        self.assertEqual(two, 2)
        self.assertIn("something1", kargs)
        self.assertTrue(kargs['something1'])


if __name__ == '__main__':
    unittest.main()
