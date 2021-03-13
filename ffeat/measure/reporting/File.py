###############################
#
# Created by Patrik Valkovic
# 3/13/2021
#
###############################


class Buffer:
    def __init__(self, file):
        self._file = file

    def __call__(self, metric, *args, **kwargs):
        print(str(metric), file=self._file)


class File(Buffer):
    def __init__(self, file):
        super().__init__(open(file, "w"))

    def __del__(self):
        self._file.close()
        del self._file
