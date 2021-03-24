import torch as t
import matplotlib.pyplot as plt

DIM = 40
LAMBDA = 1000
MU = 400
ITERS = 100
def fn(x):
    return t.sum(x ** 2, dim=1)

C = t.diag(t.ones(DIM, dtype=t.float))
m = t.tensor(t.randint(-5,5,(DIM,)), dtype=t.float)
w_m = 1 / MU  # muze byt ruzne pro kazdy prvek z MU
c_m = 0.7

for g in range(ITERS):
    plt.figure(figsize=(8, 8))
    # plot m
    plt.scatter(m[0], m[1], c='r', marker='+')
    # generate new population
    dist = t.distributions.MultivariateNormal(m, C)
    pop = dist.rsample((LAMBDA,))
    # plot population
    plt.scatter(pop[:,0], pop[:,1], c='b')
    # evaluate
    eval = fn(pop)
    print(float(t.mean(eval)))
    best_indices = t.argsort(eval)
    # calculate new m
    picked_for_mu = pop[best_indices[:MU]]
    plt.scatter(picked_for_mu[:,0], picked_for_mu[:,1], c='g')
    new_m = t.sum(picked_for_mu * w_m, dim=0)
    plt.scatter(new_m[0], new_m[1], c='r', marker='x', alpha=0.3)
    new_m = m + c_m * (new_m - m)
    plt.scatter(new_m[0], new_m[1], c='r', marker='x')
    # C empirical
    pop_mean = t.mean(pop, dim=0)
    # empirical of all the points
    #C_emp = pop - pop_mean[None,:]
    #C_emp = 1 / (LAMBDA - 1) * t.sum(C_emp[:,:,None] * C_emp[:,None,:], dim=0)
    #new_C = pop - m[None,:]
    # estimate of all the points
    #new_C = 1 / LAMBDA * t.sum(new_C[:,:,None] * new_C[:,None,:], dim=0)
    #new_C = t.zeros((2,2))
    # estimate of picked points
    #for i in range(2):
    #    for j in range(2):
    #        tmp = 0
    #        for _w in range(MU):
    #            tmp += w_m * (picked_for_mu[_w,i] - m[i]) * (picked_for_mu[_w,j] - m[j])
    #        new_C[i,j] = tmp
    new_C = picked_for_mu - m[None,:]
    new_C = t.sum(w_m * new_C[:,:,None] * new_C[:,None,:], dim=0)  # estimate of only best values

    # update estimates
    m = new_m
    C = new_C

    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.savefig(f"Z:/UDPreceive/step_{g:03d}.png")
    plt.close()
