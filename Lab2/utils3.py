import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from utils1 import get_gmm_samples


def p(x):
    return x**0.65 * np.exp(-(x**2)/2) * (x > 0)


def q(x, mu, sig):
    return 2 * norm.pdf(x, loc=mu, scale=np.sqrt(sig))


def f(x):
    return 2 * np.sin(np.pi/1.5 * x) * (x > 0)


# Poor importance sampling
def sample_importance_sampling(n_samples, f, p, q, q_params):
    # samples of q
    x = 2 * norm(q_params[0], np.sqrt(q_params[1])).rvs((n_samples))
    samples = x * (x > 0)

    # normalized importance weights
    importance_weights = p(samples) / q(samples, *q_params)
    normalized_weights = importance_weights / np.mean(importance_weights)

    estimator = np.mean(normalized_weights * f(samples))
    return estimator, normalized_weights


# Adaptive importance sampling
class PopulationMonteCarlo:
    def __init__(self, n_samples, n_clusters, dim, b, sigma2=1,  max_iters=100, tol=1e-3):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.tol = tol
        self.dim = dim
        self.b, self.sigma2 = b, sigma2
        self.reg = 1e-6  # Regularization for covariance matrices

    def init_params(self):
        self.mus = np.random.randn(self.n_clusters, self.dim)
        self.sigmas = np.zeros((self.n_clusters, self.dim, self.dim)) + np.eye(self.dim)
        self.alphas = np.ones(self.n_clusters) / self.n_clusters

    def target_nu_pdf(self, X):
        X_trans = X.copy()
        X_trans[:, 1] += self.b * (X_trans[:, 0]**2 - self.sigma2)
        pdf = multivariate_normal.pdf(X_trans, mean=np.zeros(self.dim), cov=self.sigma2 * np.eye(self.dim))
        return pdf

    def current_pdf(self, X):
        # Compute importance density q as a mixture of Gaussians
        q_values = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            q_values[:, k] = self.alphas[k] * multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k] + self.reg)
        return q_values.sum(axis=1)

    def compute_importance_weights(self, current_q, target_pdf):
        # Compute importance weights as target_pdf(X) / current_q_pdf(X)
        return target_pdf / current_q

    def e_step(self, X, weights):
        # Compute responsibilities (gammas) based on the updated weights
        gammas = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            pdf_k = multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])
            gammas[:, k] = self.alphas[k] * pdf_k * weights
        gammas /= gammas.sum(axis=1, keepdims=True)
        return gammas

    def m_step(self, X, gammas):
        n_samples, dim = X.shape
        Nk = gammas.sum(axis=0)

        # Update alphas (mixture weights)
        self.alphas = Nk / n_samples

        # Update means (mus)
        self.mus = (gammas.T @ X) / Nk[:, None]

        # Update covariances (sigmas)
        self.sigmas = np.zeros((self.n_clusters, dim, dim))
        for k in range(self.n_clusters):
            difference = X - self.mus[k]
            weighted_diff = gammas[:, k, None] * difference
            self.sigmas[k] = (weighted_diff.T @ difference) / Nk[k] + self.reg

    def fit(self):
        # Initialize parameters
        self.init_params()
        self.log_likelihoods = []

        for iter in range(self.max_iters):
            X, _ = get_gmm_samples(self.n_clusters, self.alphas, self.mus, self.sigmas, self.n_samples)

            # Step (ii): Compute importance weights
            target_pdf = self.target_nu_pdf(X)
            current_q = self.current_pdf(X)
            weights = self.compute_importance_weights(current_q, target_pdf)

            # Step (iii): E-step
            gammas = self.e_step(X, weights / weights.mean(axis=0))

            # M-step
            self.m_step(X, gammas)

            # Compute log-likelihood for convergence check
            likelihood = np.sum(np.log(current_q))
            self.log_likelihoods.append(likelihood)

            # Check for convergence
            if iter > 0 and abs(likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {iter}")
                break
        return self
