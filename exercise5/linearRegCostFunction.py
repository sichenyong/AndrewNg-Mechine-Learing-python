import numpy as np

def cost_reg(theta,x_data, y_data, lamda):
    m = x_data.shape[0]
    x_data = np.insert(x_data, 0, 1,axis=1)
    y_data = y_data.reshape((-1,1))
    theta = theta.reshape((-1,1))
    hx = x_data @ theta # m * 1
    left = np.sum(np.power(hx - y_data,2)) / 2 / m
    right = np.sum(np.power(theta[1:], 2)) * lamda / 2 / m
    return left + right

def gradient_reg(theta, x_data, y_data, lamda):
    m = x_data.shape[0]
    x_data = np.insert(x_data, 0, 1, axis=1)
    y_data = y_data.reshape((-1,1))
    theta = theta.reshape((-1,1))
    hx = x_data @ theta
    grad = (x_data.T @ (hx - y_data)) / m # m * 1
    reg_item = theta * lamda / m
    grad[1:] = grad[1:] + reg_item[1:]
    return grad.flatten()


