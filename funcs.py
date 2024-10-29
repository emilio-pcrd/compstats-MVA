import numpy as np


def sgd(X, y, w_init, lr, iters=1000):
    n = len(y)
    w = w_init

    for _ in range(iters):
        idx = np.random.randint(0, n)
        x_i, y_i = X[idx], y[idx]
        output = np.dot(x_i, w)

        w = w - 2 * lr * x_i * (output - y_i)

    return w


def random_samples(n, d):
    samples = np.random.rand(n, d)*20 - 10
    return samples
