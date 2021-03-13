###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import sys
from .File import Buffer


class Console(Buffer):
    def __init__(self, metric_name):
        super().__init__(sys.stdout)
        self.name = metric_name

    def __call__(self, metric, *args, **kwargs):
        super().__call__(f"{self.name}:{metric}", *args, **kwargs)
