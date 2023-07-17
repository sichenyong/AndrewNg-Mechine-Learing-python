import numpy as np
import cofiCostFunc
def compute_grad_numerically(params, Y, R, num_users, num_movies, num_features, l=0):
    epsilon = 1e-4
    grad = np.zeros(params.shape)
    for i in range(params.size):
        params_tr = params.copy()
        params_tr[i] += epsilon
        params_tl = params.copy()
        params_tl[i] -= epsilon
        costr = cofiCostFunc.cost(params_tr,Y,R,num_users,num_movies,num_features,l)
        costl = cofiCostFunc.cost(params_tl,Y,R,num_users,num_movies,num_features,l)
        grad[i] = (costr - costl) / (2 * epsilon)
    return grad