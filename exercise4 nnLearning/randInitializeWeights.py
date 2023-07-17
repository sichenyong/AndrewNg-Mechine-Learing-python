import numpy as np

def initialize_weights(l_in, l_out):
    epsilon_init = 0.12
    weights = np.random.rand(l_out, l_in + 1) * 2 * epsilon_init - epsilon_init
    return weights