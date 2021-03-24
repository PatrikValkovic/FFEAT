import torch as t
import matplotlib.pyplot as plt

DIM = 2
LAMBDA = 100
MU = 20
ITERS = 100
def fn(x):
    return t.sum(x ** 2, dim=1)

C = t.diag(t.ones(DIM, dtype=t.float))
m = t.tensor([3,4], dtype=t.float)
c_m = 0.7
c_mu = 0.4
w_m = 1 / MU  # muze byt ruzne pro kazdy prvek z MU
w_l = -1 / (2 * LAMBDA - 2 * MU)
w_sum = w_m * MU + w_l * (LAMBDA - MU)

for _ in range(ITERS):
    plt.figure(figsize=(8, 8))
    # generate new population
    dist = t.distributions.MultivariateNormal(m, C)
    pop = dist.rsample((LAMBDA,))
    # plot population
    plt.scatter(pop[:,0], pop[:,1], c='b')
    # evaluate
    eval = fn(pop)
    print(f"{_:03d}:{float(t.mean(eval)):.12f}")
    best_indices = t.argsort(eval)
    # calculate new m
    picked_for_mu = pop[best_indices[:MU]]
    plt.scatter(picked_for_mu[:,0], picked_for_mu[:,1], c='g')
    new_m = t.sum(picked_for_mu * w_m, dim=0)
    plt.scatter(new_m[0], new_m[1], c='r', marker='x', alpha=0.3)
    new_m = m + c_m * (new_m - m)
    plt.scatter(new_m[0], new_m[1], c='r', marker='x')
    # plot m
    plt.scatter(m[0], m[1], c='r', marker='+')

    # C estimate
    picked_C = picked_for_mu - m[None,:]
    picked_C = t.sum(w_m * picked_C[:,:,None] * picked_C[:,None,:], dim=0)  # estimate of only best values
    not_picked = pop[best_indices[MU:]]
    notpicked_C = not_picked - m[None,:]
    notpicked_C = t.sum(w_l * notpicked_C[:,:,None] * notpicked_C[:,None,:], dim=0)  # estimate of only best values
    new_C = picked_C + notpicked_C  # notpicked C can have negative values

    # update estimates
    m = new_m
    C = (1-c_mu*w_sum) * C + c_mu * new_C

    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.savefig(f"Z:/UDPreceive/step_{_:03d}.png")
    plt.close()
