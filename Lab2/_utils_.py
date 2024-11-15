"""module gathering the utilty functions for the lab2"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse


########### EXERCICE 1 #############

def gen_discrete_dist(x, prob):
    """
    generate  discrete distribution of X.
    n: number of elements
    x: list of the possible elements
    prob: list of probabilities for each element
    """
    n = len(x)
    assert 1-1e-3 <= sum(prob) <= 1+1e-3, f'sum of probabilities is: {sum(prob)}'

    # cumulative probabilities
    cum_prob = [0]
    for i in range(n):
        cum_prob.append(cum_prob[i] + prob[i])

    assert 1-1e-3 <= cum_prob[-1] <= 1+1e-3, f'cum prob is not 1: {cum_prob[-1]}'

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

    return gmm_sample, z_latent


def get_gmm_samples(m, alpha, mu, sigma, N):
    """
    generate N samples from a gaussian mixture model
    m, alpha, mu, sigma: same as gen_gmm_sample
    N: number of samples
    """
    _, dim = mu.shape
    samples, z = np.zeros((N, dim)), np.zeros(N)
    for i in range(N):
        sample = gen_gmm_sample(m, alpha, mu, sigma)
        samples[i] = sample[0]
        z[i] = sample[1]
    return samples, z


# EM Algorithm for GMM
class GaussianMixtureModel:
    def __init__(self, n_clusters, dim, max_iters=100, tol=1e-3):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.tol = tol
        self.reg = 1e-4 * np.eye(dim)

    def init_params(self, X):
        n_samples, dim = X.shape

        # Init with KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(X)
        labels = kmeans.labels_

        # Init gammas
        gammas = np.zeros((n_samples, self.n_clusters))
        gammas[np.arange(n_samples), labels] = 1

        Nk = gammas.sum(axis=0)

        # Init mus
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Init sigmas
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k]

        # Init alphas
        self.alphas = Nk / n_samples

    def e_step(self, X):
        '''
        Calcul des gammas avec des mu, sigma et alpha fixés.
        '''
        n_samples, _ = X.shape
        gammas = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            pdf_k = multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])        
            gammas[:, k] = pdf_k * self.alphas[k]

        # Normalization
        gamma_tot = gammas.sum(axis=1, keepdims=True)
        gammas /= gamma_tot
        return gammas

    def m_step(self, X, gammas):
        n_samples, dim = X.shape

        Nk = gammas.sum(axis=0)

        # Update mus
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Update sigmas
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k]

        # Update alphas
        self.alphas = Nk / n_samples

    def fit(self, X):
        '''
        Ajustement du modèle avec l'algorithme EM.
        '''
        self.init_params(X)
        self.log_likelihoods = []

        for iter in range(self.max_iters):
            # E-step
            gamma = self.e_step(X)

            # M-step
            self.m_step(X, gamma)

            # log-likelihood
            likelihood = np.sum(
                np.log(
                    np.sum(
                        [self.alphas[k] * multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])
                         for k in range(self.n_clusters)], axis=0)
                )
            )
            self.log_likelihoods.append(likelihood)

            # Convergence
            if iter > 0 and abs(likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Convergence atteinte à l'itération {iter}")
                break

        return self

    def predict_proba(self, X):
        '''
        Calcul des gammas pour chaque point de données.
        '''
        return self.e_step(X)

    def predict(self, X):
        '''
        Assignation des clusters pour chaque échantillon.
        '''
        gammas = self.predict_proba(X)
        return np.argmax(gammas, axis=1)


def compute_bic(X, gmm):
    n_samples, dim = X.shape
    n_clusters = gmm.n_clusters

    log_likelihood = gmm.log_likelihoods[-1]
    df = n_clusters * (dim + dim * (dim + 1) / 2) + (n_clusters - 1)

    bic = -2 * log_likelihood + df * np.log(n_samples)

    return bic


def plot_results(n_clusters, X, z, mu, sigma, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(n_clusters):
        ax.set_title(f'{title}')

        idx = [clust for clust, k in enumerate(z) if k == i]
        cluster_data = X[idx]
        covariances = sigma[i]
        means = mu[i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, alpha=0.5, label=f'Data from cluster {i + 1}')

        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=means, width=width, height=height, angle=angle, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)

        ax.scatter(means[0], means[1], marker='*', s=100, color='black')
        ax.grid(True)
    plt.legend()
    plt.show()


############# EXERCICE 3 #############


def p(x):
    return x**0.65 * np.exp(-(x**2)/2) * (x > 0)


def q(x, mu, sig):
    return 2 * norm.pdf(x, loc=mu, scale=np.sqrt(sig))


def f(x):
    return 2 * np.sin(np.pi/1.5 * x) * (x > 0)


# Poor importance sampling
def sample_importance_sampling(n_samples, f, p, q, q_params):
    # samples from  q
    samples = []
    while len(samples) < n_samples:
        x = 2 * norm(q_params[0], np.sqrt(q_params[1])).rvs((n_samples))
        valid_samples = x[x > 0]
        samples.extend(valid_samples)
    samples = np.array(samples[:n_samples])

    # normalized importance weights
    importance_weights = p(samples) / q(samples, *q_params)
    normalized_weights = importance_weights / np.mean(importance_weights)

    estimator = np.mean(normalized_weights * f(samples))
    return samples, estimator, normalized_weights


# Adaptive importance sampling
class PopulationMonteCarlo:
    def __init__(self, n_samples, n_clusters, dim, b, sigma2=1,  max_iters=100, tol=1e-3):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.tol = tol
        self.dim = dim
        self.b, self.sigma2 = b, sigma2
        self.reg = 1e-6

    def init_params(self):
        self.mus = np.random.randn(self.n_clusters, self.dim)
        self.sigmas = np.zeros((self.n_clusters, self.dim, self.dim)) + np.eye(self.dim)
        self.alphas = np.ones(self.n_clusters) / self.n_clusters

    def target_nu_pdf(self, X):
        X_trans = X.copy()
        X_trans[:, 1] += self.b * (X_trans[:, 0]**2 - self.sigma2**2)
        sigma = np.eye(self.dim)
        sigma[0, 0] = self.sigma2
        pdf = multivariate_normal.pdf(X_trans, mean=np.zeros(self.dim), cov=sigma)
        return pdf

    def current_pdf(self, X):
        # Compute importance density q as a mixture of Gaussians
        q_values = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            q_values[:, k] = self.alphas[k] * multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k] + self.reg)
        return q_values.sum(axis=1)

    def compute_importance_weights(self, current_q, target_pdf):
        # Compute importance weights as target_pdf / current_q_pdf
        return target_pdf / current_q

    def e_step(self, X, weights):
        # Compute gammas
        gammas = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            pdf_k = multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])
            gammas[:, k] = self.alphas[k] * pdf_k * weights
        gammas /= gammas.sum(axis=1, keepdims=True)
        return gammas

    def m_step(self, X, gammas):
        n_samples, dim = X.shape
        Nk = gammas.sum(axis=0)

        # Update alphas
        self.alphas = Nk / n_samples

        # Update mus
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Update sigmas
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k]

    def fit(self):
        # Initialize parameters
        self.init_params()
        self.log_likelihoods = []

        for iter in range(self.max_iters):
            X, _ = get_gmm_samples(self.n_clusters, self.alphas, self.mus, self.sigmas, self.n_samples)

            # importance weights
            target_pdf = self.target_nu_pdf(X)
            current_q = self.current_pdf(X)
            weights = self.compute_importance_weights(current_q, target_pdf)

            # E-step
            gammas = self.e_step(X, weights / weights.mean(axis=0))

            # M-step
            self.m_step(X, gammas)

            # log-likelihood
            likelihood = np.sum(np.log(current_q))
            self.log_likelihoods.append(likelihood)

            # convergence
            if iter > 0 and abs(likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {iter}")
                break
        return self
