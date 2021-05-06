###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################
import os

class Buffer:
    """
    Log the metric using file-like description.
    """
    def __init__(self, file):
        """
        Log the metric using file-like description.
        :param file: File-like description where to log the metric.
        """
        self._file = file

    def __call__(self, metric, *args, **kwargs):
        """
        Log the metric using file-like description.
        :param metric: Metric value to log.
        :param args: Arguments ignored.
        :param kwargs: Keyword arguments ignored.
        :return: None
        """
        print(str(metric), file=self._file)


class File(Buffer):
    """
    Log the metric into file.
    """
    def __init__(self, file: str):
        """
        Log the metric into file.
        :param file: Path to the file.
        """
        if "/" in file:
            slash_pos = file.rfind('/')
            dir = file[:slash_pos]
            os.makedirs(dir, exist_ok=True)
        super().__init__(open(file, "w"))

    def __del__(self):
        """
        Flush the content and close the file.
        :return: None
        """
        self._file.close()
        del self._file
