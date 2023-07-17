from scipy.special import expit
import numpy as np
def sigmoid_gradient(z):
    return np.multiply(expit(z), 1 - expit(z))