"""module gathering the utilty functions for the lab2"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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

        # Initialisation avec KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(X)
        labels = kmeans.labels_

        # Initialisation des gammas
        gammas = np.zeros((n_samples, self.n_clusters))
        gammas[np.arange(n_samples), labels] = 1

        Nk = gammas.sum(axis=0)

        # Initialisation des moyennes
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Initialisation des covariances
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k]

        # Initialisation des poids (alphas)
        self.alphas = Nk / n_samples

    def e_step(self, X):
        '''
        Calcul des gammas avec des mu, sigma et alpha fixés.
        '''
        n_samples, _ = X.shape
        gammas = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            # Calcul de la densité de probabilité pour chaque cluster
            pdf_k = multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])        
            gammas[:, k] = pdf_k * self.alphas[k]

        # Normalisation des gammas
        gamma_tot = gammas.sum(axis=1, keepdims=True)
        gammas /= gamma_tot
        return gammas

    def m_step(self, X, gammas):
        n_samples, dim = X.shape

        # Calcul de Nk
        Nk = gammas.sum(axis=0)

        # Mise à jour des moyennes
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Mise à jour des covariances
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k]

        # Mise à jour des poids
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

            # Calcul de la log-vraisemblance
            likelihood = np.sum(
                np.log(
                    np.sum(
                        [self.alphas[k] * multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])
                         for k in range(self.n_clusters)], axis=0)
                )
            )
            self.log_likelihoods.append(likelihood)

            # Vérification de la convergence
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
        # Scatter plot for each cluster
        idx = [clust for clust, k in enumerate(z) if k == i]
        cluster_data = X[idx]
        covariances = sigma[i]
        means = mu[i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, alpha=0.5, label=f'Data from cluster {i + 1}')

        # Plot the Gaussian Ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=means, width=width, height=height, angle=angle, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)
        
        # Plot cluster center
        ax.scatter(means[0], means[1], marker='*', s=100, color='black')
        ax.grid(True)
    plt.legend()
    plt.show()
