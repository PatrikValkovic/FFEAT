###############################
#
# Created by Patrik Valkovic
# 3/12/2021
#
###############################
from ffeat._common.Evaluation import Evaluation, EvaluationIndividually


class EvaluationWrapper:
    Evaluation = Evaluation
    RowEval = EvaluationIndividually
