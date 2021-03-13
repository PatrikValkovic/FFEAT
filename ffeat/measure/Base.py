###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################

class Base:
    def __init__(self, reporter):
        self.__report = reporter

    def _report(self, value):
        if self.__report is not None:
            self.__report(value)
