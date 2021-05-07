###############################
#
# Created by Patrik Valkovic
# 5/7/2021
#
###############################

class Array:
    """
    Log the metric into array.
    """
    def __init__(self, arr = None):
        """
        Log the metric into array.
        :param arr: Optional array to log metrics into. If None, the array will be created and is accessible in
        `measurements` property.
        """
        self._measurements = arr
        if self._measurements is None:
            self._measurements = []

    @property
    def measurements(self):
        """
        Get the measurements.
        :return: Measurements.
        """
        return self._measurements

    def __call__(self, metric, *args, **kwargs):
        """
        Log the metric using into array.
        :param metric: Metric value to log.
        :param args: Arguments ignored.
        :param kwargs: Keyword arguments ignored.
        :return: None
        """
        self._measurements.append(metric)
