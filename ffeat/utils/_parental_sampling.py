###############################
#
# Created by Patrik Valkovic
# 4/17/2021
#
###############################
import torch as t


def randint(popsize, samples, parents_per_sample, device):
    return t.randint(popsize, (samples, parents_per_sample), device=device, dtype=t.long)


def multinomial(popsize, samples, parents_per_sample, device):
    probs = t.tensor(1 / popsize, device=device, dtype=t.float)
    probs.as_strided_((samples,popsize), (0,0))
    sample = t.multinomial(probs, parents_per_sample, replacement=False)
    return sample
