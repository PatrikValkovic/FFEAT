###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import unittest
import torch as t
import ffeat.measure as m
import os


class ReportingTest(unittest.TestCase):
    def test_should_not_report(self):
        q = m.FitnessMedian()
        f = t.randn((1000,))
        (nf,), kargs = q(f)

    def test_should_report_to_console(self):
        q = m.FitnessMedian(m.reporting.Console('median'))
        f = t.randn((1000,))
        (nf,), kargs = q(f)

    def test_should_report_to_file(self):
        filename = './median_fitness.txt'
        q = m.FitnessMedian(m.reporting.File(filename))
        f = t.randn((1000,))
        (nf,), kargs = q(f)
        del q
        self.assertTrue(os.path.exists(filename))
        with open(filename) as file:
            val = float(file.readline())
            self.assertLess(abs(val - float(t.median(f))), 0.01)
        os.remove(filename)

    def test_should_report_file_more_times(self):
        filename = './median_fitness.txt'
        q = m.FitnessMedian(m.reporting.File(filename))
        f1 = t.randn((1000,))
        f2 = t.randn((1000,))
        f3 = t.randn((1000,))
        q(f1)
        q(f2)
        q(f3)
        del q
        self.assertTrue(os.path.exists(filename))
        with open(filename) as file:
            val = float(file.readline())
            self.assertLess(abs(val - float(t.median(f1))), 0.01)
            val = float(file.readline())
            self.assertLess(abs(val - float(t.median(f2))), 0.01)
            val = float(file.readline())
            self.assertLess(abs(val - float(t.median(f3))), 0.01)
        os.remove(filename)

    def test_should_report_to_directory(self):
        filename = './my_measure/median_fitness.txt'
        q = m.FitnessMedian(m.reporting.File(filename))
        f = t.randn((1000,))
        (nf,), kargs = q(f)
        del q
        self.assertTrue(os.path.exists(filename))
        with open(filename) as file:
            val = float(file.readline())
            self.assertLess(abs(val - float(t.median(f))), 0.01)
        os.remove(filename)
        os.rmdir("./my_measure")


if __name__ == '__main__':
    unittest.main()
