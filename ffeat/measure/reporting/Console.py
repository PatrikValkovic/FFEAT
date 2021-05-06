###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import sys
from .File import Buffer


class Console(Buffer):
    """
    Logs metric on the standard output.
    """
    def __init__(self, metric_name):
        """
        Logs metric on the standard output.
        :param metric_name: Name under which to log the metric.
        """
        super().__init__(sys.stdout)
        self.name = metric_name

    def __call__(self, metric, *args, **kwargs):
        """
        Log the metric on the standard output.
        :param metric: Metric value to log.
        :param args: Arguments ignored.
        :param kwargs: Keyword arguments ignored.
        :return: None
        """
        super().__call__(f"{self.name}:{metric}", *args, **kwargs)
