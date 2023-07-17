import numpy as np
import sigmoid
def cost_function(theta, x_data,y_data):
    theta = theta.reshape((-1,1))
    m = len(y_data)
    data = np.dot(x_data,theta)
    hx = sigmoid.sigmoid(data)
    return -1 * (np.sum(np.dot(y_data.T,np.log(hx)) + np.dot((1 - y_data).T, np.log(1 - hx)) )) / m

def gradient(theta, x_data, y_data):
    theta = theta.reshape((-1,1))
    m = len(y_data)
    data = np.dot(x_data,theta)
    hx = sigmoid.sigmoid(data)
    return np.dot(x_data.T, (hx - y_data)) / m
