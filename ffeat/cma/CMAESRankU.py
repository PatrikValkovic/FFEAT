import torch as t
import matplotlib.pyplot as plt

DIM = 2
LAMBDA = 100
MU = 4
ITERS = 100
def fn(x):
    return t.sum(x ** 2, dim=1)

C = t.diag(t.ones(DIM, dtype=t.float))
m = t.tensor([3,4], dtype=t.float)
w_m = 1 / MU  # muze byt ruzne pro kazdy prvek z MU
c_m = 0.1
c_mu = 0.1

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
    new_C = picked_for_mu - m[None,:]
    new_C = t.sum(w_m * new_C[:,:,None] * new_C[:,None,:], dim=0)  # estimate of only best values

    # update estimates
    m = new_m
    C = (1-c_mu) * C + c_mu * new_C

    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.savefig(f"Z:/UDPreceive/step_{_:03d}.png")
    plt.close()
