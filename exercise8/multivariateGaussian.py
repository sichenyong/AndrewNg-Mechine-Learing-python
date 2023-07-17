import numpy as np

def do(X, mu, sigma2):
    k = len(mu)
    if (sigma2.shape[0] == 1) or (sigma2.shape[1] == 1):
        sigma2 = np.diag(sigma2.flatten())
    X = X - mu.reshape((1, -1))
    px = np.exp(-0.5 * np.sum(np.multiply(X @ np.linalg.pinv(sigma2), X),axis=1))
    px /= np.linalg.det(sigma2)**(0.5)
    px /= np.power(2 * np.pi, k / 2)
    return px