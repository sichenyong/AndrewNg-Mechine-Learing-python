import numpy as np

def cost(params, Y, R, num_users, num_movies, num_features, lamda=0):
    X = params[0:num_movies*num_features].reshape((num_movies,-1))
    Theta = params[num_movies*num_features:].reshape((num_users, -1))
    # m, n = Y.shape
    # cost = 0
    # for i in range(m):
    #     for j in range(n):
    #         if R[i,j] == 1:
    #             cost += np.power(Theta[j,:] @ X[i,:].T - Y[i,j] , 2)
    # return cost /2
    diff = X @ Theta.T - Y
    diff = np.power(diff * R , 2)
    cost = np.sum(diff) / 2
    reg = np.sum(np.power(Theta,2)) + np.sum(np.power(X,2))
    return cost + reg * lamda / 2

def gradient(params, Y, R, num_users, num_movies, num_features, lamda=0):
    # X 5 * 3
    X = params[0:num_movies*num_features].reshape((num_movies,-1))
    # Theta 4 * 3
    Theta = params[num_movies*num_features:].reshape((num_users, -1))
    diff = X @ Theta.T - Y
    # diff 5 * 4
    diff = np.multiply(diff, R)
    X_grad = diff @ Theta + lamda * X
    Theta_grad = diff.T @ X + lamda * Theta
    return np.hstack((X_grad.flatten(), Theta_grad.flatten()))

