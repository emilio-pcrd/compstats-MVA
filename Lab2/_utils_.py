"""module gathering the utilty functions for the lab2"""

import numpy as np


def gen_discrete_dist(x, prob):
    """
    generate  discrete distribution of X.
    n: number of elements
    x: list of the possible elements
    prob: list of probabilities for each element
    """
    n = len(x)
    assert sum(prob) == 1, f'sum of probabilities is: {sum(prob) +- 1e-6}'

    # compute cumulative probabilities
    cum_prob = [0]
    for i in range(n):
        cum_prob.append(cum_prob[i] + prob[i])

    assert cum_prob[-1] == 1, f'cum prob is not 1: {cum_prob[-1]}'

    # generate the distribution
    u = np.random.rand()
    for i in range(n):
        if cum_prob[i] <= u <= cum_prob[i+1]:
            return x[i]


def get_samples(x, prob, N):
    """
    generate N samples of the discrete distribution
    x, prob: same as gen_discrete_dist
    N: number of samples
    """
    samples = []
    for _ in range(N):
        samples.append(gen_discrete_dist(x, prob))

    return samples


# gaussian mixture model
def gen_gmm_sample(m, alpha, mu, sigma):
    """
    generate a sample from a gaussian mixture model
    m: number of components
    alpha: list of probabilities for each component
    mu: list of means for each component
    sigma: list of standard deviations for each component
    """
    # latent variable with exo1 function
    z_latent = gen_discrete_dist(list(range(m)), prob=alpha)

    # gmm_sample
    if mu[0].shape != ():
        gmm_sample = np.random.multivariate_normal(
                                                mu[z_latent],
                                                sigma[z_latent]
                                                 )
    else:
        gmm_sample = np.random.normal(mu[z_latent], sigma[z_latent])

    return gmm_sample


def get_gmm_samples(m, alpha, mu, sigma, N):
    """
    generate N samples from a gaussian mixture model
    m, alpha, mu, sigma: same as gen_gmm_sample
    N: number of samples
    """
    samples = []
    for _ in range(N):
        samples.append(gen_gmm_sample(m, alpha, mu, sigma))

    return samples


# EM algorithm
def em_gmm(X, m, max_iters=100, eps=1e-6):
    """
    EM algorithm for gaussian mixture model
    X: data
    m: int, number of components
    max_iters: int, maximum number of iterations
    eps: float, convergence criterion
    """
    n, d = np.array(X).shape

    # initialize parameters
    alpha = np.ones(m) / m
    mu = [np.random.rand(d) for _ in range(m)]
    sigma = [np.eye(d) for _ in range(m)]

    gamma = np.zeros((n, m))
    log_likelihoods = []

    # EM algorithm
    for iters in range(max_iters):
        # E-step
        for i in range(n):
            for j in range(m):
                in_e = -0.5 * (X[i] - mu[j]).T @ np.linalg.inv(sigma[j]) @ (X[i] - mu[j])
                gamma[i, j] = alpha[j] * np.exp(in_e)
            gamma[i, :] = gamma[i, :] / sum(gamma[i])

        # M-step -> update parameters
        alpha = np.mean(gamma, axis=0)
        for j in range(m):
            mu[j] = sum([gamma[i, j] * X[i] for i in range(n)]) / sum(gamma[:, j])
            sigma[j] = sum([gamma[i, j] * np.outer(X[i] - mu[j], X[i] - mu[j]) for i in range(n)]) / sum(gamma[:, j])

        # log-likelihood calculation
        log_likelihood = np.sum(np.log(np.dot(gamma, alpha)))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iters > 0 and abs(log_likelihood - log_likelihoods[-2]) < eps:
            break

    return alpha, mu, sigma, log_likelihoods
