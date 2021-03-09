###############################
#
# Created by Patrik Valkovic
# 3/9/2021
#
###############################
from typing import Tuple, Dict, Any

class Pipe:
    def __call__(self, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        return tuple(args), kwargs