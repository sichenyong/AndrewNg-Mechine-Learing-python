import numpy as np

def feature_scaling(x_data):
    mean = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)
    result = (x_data - mean) / std
    return result, mean, std