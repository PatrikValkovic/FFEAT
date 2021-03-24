import torch as t
import matplotlib.pyplot as plt
import math

DIM = 40
LAMBDA = 1000  # int(4 + math.floor(3 * math.log(DIM)))
MU = 400  # LAMBDA // 2
ITERS = 100
def fn(x):
    return t.sum(x ** 2, dim=1)


# MEAN
m = t.randint(-5,5,(DIM,))
c_m = 0.7  # learning rate for mean

# COVARIANCE
B = t.eye(DIM)
D = t.eye(DIM)
C = B @ D @ D @ B.T
c_1 = 2 / DIM ** 2
w_m = 1 / MU  # weights for best individuals
mu_eff = 1 / (w_m ** 2 * MU)
c_mu = min(1-c_1, mu_eff / DIM ** 2)
w_l = -1 / (2 * LAMBDA - 2 * MU)  # weight for not picked individuals
w_sum = w_m * MU + w_l * (LAMBDA - MU)
p_c = t.zeros(DIM)
c_c = (4 + mu_eff / DIM) / (DIM + 4 + 2 * mu_eff / DIM)

# STEP SIZE
sigma = 1.0
p_sigma = t.zeros(DIM)
c_sigma = (mu_eff + 2) / (DIM + mu_eff + 5)
d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (DIM + 1))) + c_sigma
E_N01 = math.sqrt(DIM) * (1 - 1/4/DIM + 1 / 21 / DIM**2)


for g in range(ITERS):
    plt.figure(figsize=(8, 8))

    # generate new population
    dist = t.distributions.MultivariateNormal(t.zeros(DIM), t.eye(DIM))
    z = dist.rsample([LAMBDA])
    y = z @ B.T @ D
    x = m[None,:] + sigma * y
    # plot population
    plt.scatter(x[:,0], x[:,1], c='b')

    # evaluate
    eval = fn(x)
    print(f"{g:03d}:{float(t.mean(eval)):.12f}")
    best_indices = t.argsort(eval)

    # Selection and recombination
    x_picked = x[best_indices[:MU]]
    plt.scatter(x_picked[:,0], x_picked[:,1], c='g')
    y_picked = y[best_indices[:MU]]
    y_w = t.sum(y_picked * w_m, dim=0)
    new_m = m + c_m * sigma * y_w
    plt.scatter(m[0], m[1], c='r', marker='+')
    plt.scatter(m[0]+y_w[0], m[1] + y_w[1], c='r', marker='x', alpha=0.3)
    plt.scatter(new_m[0], new_m[1], c='r', marker='x')
    m = new_m

    # Step size control
    C_power_negative_half = B @ t.diag(1 / t.diag(D)) @ B.T
    print(f"Previous p_sigma: {p_sigma}")
    p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_power_negative_half @ y_w
    print(f"New p_sigma: {p_sigma}")
    print(f"Previous sigma: {sigma}")
    sigma = sigma * t.exp(c_sigma / d_sigma * (t.norm(p_sigma) / E_N01 - 1))
    print(f"New sigma: {sigma}")

    # Covariance matrix adaptation
    h_sigma = 0.0
    if t.norm(p_sigma) / math.sqrt(1-(1-c_sigma)**(2*g+1)) < (1.4 + 2/(DIM + 1)) * E_N01:
        h_sigma = 1.0
    print(f"Old p_c: {p_c}")
    p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c)*mu_eff) * y_w
    print(f"New p_c: {p_c}")
    weights = t.full((LAMBDA,), w_l)
    weights.multiply_(DIM / t.pow(t.norm(y @ C_power_negative_half.T, dim=1), 2))
    weights[best_indices[:MU]] = w_m
    print(f"Old C\n{C}")
    delta_h_sigma = float((1 - h_sigma) * c_c * (2 - c_c) <= 1)
    C = (1 + c_1 * delta_h_sigma - c_1 - c_mu * w_sum) * C + \
        c_1 * t.outer(p_sigma, p_sigma) + \
        c_mu * t.sum(weights[:,None,None] * y[:,:,None] * y[:,None,:], dim=0)
    print(f"New C\n{C}")
    print(f"Old B\n{B}\nOld D\n{D}")
    D, B = t.symeig(C, eigenvectors=True)
    D = D.sqrt().diag_embed()
    print(f"New B\n{B}\nNew D\n{D}")

    #picked_C = picked_for_mu - m[None,:]
    #picked_C = t.sum(w_m * picked_C[:,:,None] * picked_C[:,None,:], dim=0)  # estimate of only best values
    #not_picked = pop[best_indices[MU:]]
    #notpicked_C = not_picked - m[None,:]
    #notpicked_C = t.sum(w_l * notpicked_C[:,:,None] * notpicked_C[:,None,:], dim=0)  # estimate of only best values
    #new_C = picked_C + notpicked_C  # notpicked C can have negative values

    # update estimates
    #m = new_m
    #C = (1-c_mu*w_sum) * C + c_mu * new_C

    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.savefig(f"Z:/UDPreceive/step_{g:03d}.png")
    plt.close()
