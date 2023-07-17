import numpy as np

def estimate(x_data):
    m,n = x_data.shape
    mu = np.zeros((1,n))
    sigma2 = np.zeros((1,n))
    for i in range(m):
        mu += x_data[i,:]
    mu /= m

    for i in range(m):
        sigma2 += np.power(x_data[i,:] - mu,2)
    
    sigma2 /= (m - 1)
    return mu, sigma2




