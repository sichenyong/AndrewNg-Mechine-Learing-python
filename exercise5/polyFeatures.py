import numpy as np

def mapFeatures(x_data,P):
    m = x_data.shape[0]
    X_poly = np.zeros((m, P))
    for i in range(P):
        X_poly[:, i] = np.power(x_data[:,0],i + 1)
    return X_poly

    
    