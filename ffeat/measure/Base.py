###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################

class Base:
    """
    Base class for the measurements.
    """
    def __init__(self, reporter):
        """
        Base class for the measurements.
        :param reporter: Reporter to which report the results. May be None and then the values are just passed in the
        keyword arguments.
        """
        self.__report = reporter

    def _report(self, value):
        """
        Report the value to the reporter if set.
        :param value: Value to report
        :return: None
        """
        if self.__report is not None:
            self.__report(value)
