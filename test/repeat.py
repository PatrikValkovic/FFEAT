###############################
#
# Created by Patrik Valkovic
# 3/20/2021
#
###############################
def repeat(max_repeat: int):
    def _fn_o(func):
        def _fn(*args, **kwargs):
            for i in range(max_repeat):
                try:
                    func(*args, **kwargs)
                    break
                except AssertionError as e:
                    print('f', end="")
                    if i == max_repeat - 1:
                        raise e
        return _fn
    return _fn_o
