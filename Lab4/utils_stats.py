import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal, invgamma


def target_distribution(x, y, a):
    return np.exp(-(x**2)/(a**2) - y**2 - 1/4*((x**2)/(a**2) - y**2)**2)


def sample_P1(x, y, a, sigma_x):
    x_proposed = np.random.normal(x, sigma_x)
    current_prob = target_distribution(x, y, a)
    proposed_prob = target_distribution(x_proposed, y, a)

    alpha_ratio = min(1, proposed_prob / current_prob)
    if np.random.rand() < alpha_ratio:
        return x_proposed, y, alpha_ratio, True
    else:
        return x, y, alpha_ratio, False


def sample_P2(x, y, a, sigma_y):
    y_proposed = np.random.normal(y, sigma_y)
    current_prob = target_distribution(x, y, a)
    proposed_prob = target_distribution(x, y_proposed, a)

    alpha_ratio = min(1, proposed_prob / current_prob)
    if np.random.rand() < alpha_ratio:
        return x, y_proposed, alpha_ratio, True
    else:
        return x, y, alpha_ratio, False


def sample_2d(x, y, sigma_x, sigma_y, a):
    proposal_x, y, ratio_x, bool_accept = sample_P1(x, y, a, sigma_x)
    if bool_accept:
        accept_attempt_x = 1
    else:
        accept_attempt_x = 0
    x, proposal_y, ratio_y, bool_accept = sample_P2(proposal_x, y, a, sigma_y)
    if bool_accept:
        accept_attempt_y = 1
    else:
        accept_attempt_y = 0

    return proposal_x, proposal_y, ratio_x, ratio_y, accept_attempt_x, accept_attempt_y


def MHGibbs(x0, y0, sigma_x, sigma_y, a, n):
    samples = []
    x, y = x0, y0

    accept_attempt_x, accept_attempt_y = 0, 0
    accepted_x, accepted_y = [], []

    for i in range(1, n + 1):
        x, y, ratio_x, ratio_y, acc_x, acc_y = sample_2d(x, y, sigma_x, sigma_y, a)
        samples.append([x, y])

        accept_attempt_x += acc_x
        accept_attempt_y += acc_y

        if i % 50 == 0:
            accepted_x.append(accept_attempt_x / 50)
            accepted_y.append(accept_attempt_y / 50)

            accept_attempt_x = 0
            accept_attempt_y = 0

    print(f'Mean Acceptance rate x: {np.mean(accepted_x)}')
    print(f'Mean Acceptance rate y: {np.mean(accepted_y)}')

    return np.array(samples), np.array([accepted_x, accepted_y])


# 1.B.1
def Adaptive_MHG(x0, y0, a, n_batch, batch_size):
    samples = []
    x, y = x0, y0
    L = np.zeros(2)
    accept_rates = np.zeros((n_batch, 2))

    for j in tqdm(range(n_batch)):
        accept_attempt_x, accept_attempt_y = 0, 0

        for _ in range(batch_size):
            sigma_x = np.exp(L[0])
            sigma_y = np.exp(L[1])
            x, y, _, _, acc_x, acc_y = sample_2d(x, y, sigma_x, sigma_y, a)
            samples.append([x, y])
            accept_attempt_x += acc_x
            accept_attempt_y += acc_y

        accept_rates[j, 0] = accept_attempt_x / batch_size
        accept_rates[j, 1] = accept_attempt_y / batch_size

        delta = min(0.01, 1 / np.sqrt(j + 1))
        if accept_rates[j, 0] > 0.234:
            L[0] += delta
        else:
            L[0] -= delta

        if accept_rates[j, 1] > 0.234:
            L[1] += delta
        else:
            L[1] -= delta

    return np.array(samples), accept_rates, L


# 1.B.2
def target_banana(x, B=0.03):
    return np.exp(-(x[0]**2)/200 - 0.5 * (x[1] + B*(x[0]**2) - 100*B)**2-0.5*(np.sum(x[2:]**2)))


def update_sample(x, j, L, B=0.03):
    update = np.random.normal(x[j], np.exp(L[j]))
    proposal = np.concatenate((x[:j], [update], x[j + 1:]))
    alpha = min(1, target_banana(proposal, B) / target_banana(x, B))
    accepted = np.random.rand() < alpha

    return (proposal if accepted else x), accepted


def Adaptive_HM_banana(n_samples, d, initial_state, proba, B=0.03):
    x = initial_state
    L = np.zeros(d)
    samples = np.zeros((n_samples, d))
    accepted_x = np.zeros((n_samples // 50, d))
    batch_accepted_x = np.zeros(d)

    for i in tqdm(range(n_samples)):
        for j in range(d):
            if np.random.uniform() < proba[j]:
                x, accepted = update_sample(x, j, L, B)
                if accepted:
                    batch_accepted_x[j] += 1
        samples[i, :] = x

        if i % 50 == 0 and i > 0:
            k = i // 50
            accepted_x[k, :] = batch_accepted_x / 50
            batch_accepted_x = np.zeros(d)
            delta = min(0.01, k ** -0.5)
            for j in range(d):
                L[j] += delta if accepted_x[k, j] >= 0.234 else -delta

    return samples, accepted_x


#Exo 2
def target_gaussian_density(x, w, mu, sigma):
    density = 0
    for i in range(20):
        density += w * multivariate_normal(mean=mu[i], cov=np.ones(2)*sigma).pdf(x)
    return  density


def MH_SRW(n_samples, init, w, mu, sigma):
    samples = np.zeros((n_samples, 2))
    samples[0, :] = init
    for i in tqdm(range(n_samples)):
        sam = np.random.normal(samples[i-1, 0], 1)
        prop = np.array([sam, samples[i-1, 1]])
        current = samples[i-1, :]
        alpha = min(1, target_gaussian_density(prop, w, mu, sigma) / target_gaussian_density(current, w, mu, sigma))
        if np.random.rand() < alpha:
            samples[i, 0] = sam
        else:
            samples[i, 0] = samples[i-1, 0]

        sam = np.random.normal(samples[i-1, 1], 1)
        prop = np.array([samples[i, 0], sam])
        current = np.array([samples[i, 0], samples[i-1, 1]])
        alpha = min(1, target_gaussian_density(prop, w, mu, sigma) / target_gaussian_density(current, w, mu, sigma))
        if np.random.rand() < alpha:
            samples[i, 1] = sam
        else:
            samples[i, 1] = samples[i-1, 1]
    return samples


def Adaptative_MH_RW(n_samples, init, w, mu, sigma):
    samples = np.zeros((n_samples, 2))
    samples[0, :] = init
    L = np.zeros(2)
    acc_rate_x = 0
    acc_rate_y = 0
    for i in tqdm(range(n_samples)):
        sam = np.random.normal(samples[i-1, 0], np.exp(L[0]))
        prop = np.array([sam, samples[i-1, 1]])
        current = samples[i-1, :]
        alpha = min(1, target_gaussian_density(prop, w, mu, sigma) / target_gaussian_density(current, w, mu, sigma))
        if np.random.rand() < alpha:
            samples[i, 0] = sam
            acc_rate_x += 1
        else:
            samples[i, 0] = samples[i-1, 0]

        sam = np.random.normal(samples[i-1, 1], np.exp(L[1]))
        prop = np.array([samples[i, 0], sam])
        current = np.array([samples[i, 0], samples[i-1, 1]])
        alpha = min(1, target_gaussian_density(prop, w, mu, sigma) / target_gaussian_density(current, w, mu, sigma))
        if np.random.rand() < alpha:
            samples[i, 1] = sam
            acc_rate_y += 1
        else:
            samples[i, 1] = samples[i-1, 1]

        if i % 50 == 0 and i > 0:
            j = i // 50
            if acc_rate_x/50 > 0.24:
                L[0] += min(0.01, 1/np.sqrt(j))
            else:
                L[0] -= min(0.01, 1/np.sqrt(j))

            if acc_rate_y/50 > 0.24:
                L[1] += min(0.01, 1/np.sqrt(j))
            else:
                L[1] -= min(0.01, 1/np.sqrt(j))

            acc_rate_x = 0
            acc_rate_y = 0
    return samples


# Parallel Tempering
def Parrallel_Tempering(n_samples, temperatures, init, w, mu, sigma):
    samples = np.zeros((len(temperatures), n_samples, 2))
    samples[:, 0, :] = init
    for n in tqdm(range(n_samples)):
        for t in range(len(temperatures)):
            sam = np.random.normal(samples[t, n-1, 0], (0.25 * np.sqrt(temperatures[t])))
            prop = np.array([sam, samples[t, n-1, 1]])
            current = samples[t, n-1, :]
            alpha = min(1, np.power(target_gaussian_density(prop, w, mu, sigma), 1/temperatures[t]) / np.power(target_gaussian_density(current, w, mu, sigma), 1/temperatures[t]))

            if np.random.rand() < alpha:
                samples[t, n, 0] = sam
            else:
                samples[t, n, 0] = samples[t, n-1, 0]

            sam = np.random.normal(samples[t, n-1, 1], (0.25 * np.sqrt(temperatures[t])))
            prop = np.array([samples[t, n, 0], sam])
            current = np.array([samples[t, n, 0], samples[t, n-1, 1]])
            alpha = min(1, target_gaussian_density(prop, w, mu, sigma) / target_gaussian_density(current, w, mu, sigma))
            if np.random.rand() < alpha:
                samples[t, n, 1] = sam
            else: samples[t, n, 1] = samples[t, n-1, 1]

        i = np.random.randint(0, len(temperatures))
        if i == 0:
            j = i+1
        elif i == len(temperatures)-1:
            j = i-1
        else:
            j = np.random.choice([i-1, i+1])
        num = np.power(target_gaussian_density(samples[j, n, :], w, mu, sigma), 1/temperatures[i]) * np.power(target_gaussian_density(samples[i, n, :], w, mu, sigma), 1/temperatures[j])
        den = np.power(target_gaussian_density(samples[i, n, :], w, mu, sigma), 1/temperatures[i]) * np.power(target_gaussian_density(samples[j, n, :], w, mu, sigma), 1/temperatures[j])
        alpha_swap = min(1, num / den)

        if np.random.rand() < alpha_swap:
            res = samples[i, n, :]
            samples[i, n, :] = samples[j, n, :]
            samples[j, n, :] = res

    return samples


# Exercice 3, gibb sampler
def gibbs(init_sigma, init_tau, init_mu, init_X, Y, N, k, beta, alpha, gamma,  n_samples):
    sigmas = np.zeros((n_samples))
    taus = np.zeros((n_samples))
    mus = np.zeros((n_samples))
    store_X = np.zeros((n_samples, N))
    mu = init_mu
    tau = init_tau
    sigma = init_sigma

    X = init_X
    for i in tqdm(range(n_samples)):
        sigma = invgamma.rvs(N/2 + alpha, scale=(beta + np.sum((X - mu)**2)/2))
        tau = invgamma.rvs(N*k/2 + gamma, scale=(beta + np.sum((Y.T - X)**2)/2))
        mu = np.random.normal(np.mean(X), scale=sigma/N)

        X = np.zeros(N)
        for j in range(N):
            var = sigma * tau / (k*sigma + tau)
            mean = (tau * mu + sigma * np.sum(Y[j, :])) / (k*sigma + tau)
            X[j] = np.random.normal(mean, scale=var)

        sigmas[i] = sigma
        taus[i] = tau
        mus[i] = mu
        store_X[i, :] = X
    return sigmas, taus, mus, store_X


def block_gibbs(init_sigma, init_tau, init_mu, init_X, Y, N, k, beta, alpha, gamma, num_samples):
    store_sigma = np.zeros(num_samples)
    store_tau = np.zeros(num_samples)
    store_mu = np.zeros(num_samples)
    store_X = np.zeros((num_samples, N))

    mu = init_mu
    tau = init_tau
    sigma = init_sigma
    X = init_X

    for i in tqdm(range(num_samples)):
        sigma = invgamma.rvs(N/2 + alpha, scale=(beta + np.sum((X - mu)**2)/2))
        tau = invgamma.rvs(N*k/2 + gamma, scale=(beta + np.sum((Y - X[:, None])**2)/2))
        precision = np.zeros((N+1, N+1))

        for j in range(N):
            precision[j, j] = k/tau + 1/sigma
            precision[j, N] = -1/sigma
            precision[N, j] = -1/sigma
        precision[N, N] = N/sigma

        mean = np.zeros(N + 1)
        for j in range(N):
            mean[j] = np.sum(Y[j, :])/tau

        mean[N] = 0
        cov = np.linalg.inv(precision)
        mean = cov @ mean
        joint_sample = np.random.multivariate_normal(mean, cov)

        X = joint_sample[:N]
        mu = joint_sample[N]

        store_sigma[i] = sigma
        store_tau[i] = tau
        store_mu[i] = mu
        store_X[i] = X

    return store_sigma, store_tau, store_mu, store_X
