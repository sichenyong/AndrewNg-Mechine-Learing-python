import numpy as np
import sigmoid
def cost_function(theta, x_data,y_data, lamda):
    theta = theta.reshape((-1,1))
    m = len(y_data)
    data = np.dot(x_data,theta)
    hx = sigmoid.sigmoid(data)
    left = -1 * (np.sum(np.dot(y_data.T,np.log(hx)) + np.dot((1 - y_data).T, np.log(1 - hx)) )) / m
    reg_item = lamda * np.sum(np.power(theta[1:,:],2)) / 2 / m
    return left + reg_item

def gradient(theta, x_data, y_data, lamda):
    theta = theta.reshape((-1,1))
    m = len(y_data)
    data = np.dot(x_data,theta)
    hx = sigmoid.sigmoid(data)
    left = np.dot(x_data.T, (hx - y_data)) / m
    reg_item = np.multiply(lamda,theta) / m
    # 不对theta0正则化
    reg_item[0:1, :] = 0
    return left + reg_item
