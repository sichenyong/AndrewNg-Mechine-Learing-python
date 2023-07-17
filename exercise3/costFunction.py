import numpy as np

# gz
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, x_data, y_data,lamda):
    theta = theta.reshape((-1, 1))
    y_data = y_data.reshape((-1, 1))
    hx = sigmoid(np.dot(x_data, theta))
    ln_h = np.log(hx)
    left = -np.dot(ln_h.T,y_data) / x_data.shape[0]
    right = -np.dot(np.log(1 - hx).T, 1 - y_data) / x_data.shape[0]
    reg = lamda * np.sum(np.power(theta[1:,:],2)) / 2 / x_data.shape[0]
    return (left + right + reg).flatten()

def gradient(theta, x_data, y_data,lamda):
    theta = theta.reshape((-1,1))
    m = len(y_data)
    data = np.dot(x_data,theta)
    hx = sigmoid(data)
    left = np.dot(x_data.T, (hx - y_data)) / m
    reg_item = np.multiply(lamda,theta) / m
    # 不对theta0正则化
    reg_item[0:1, :] = 0
    return (left + reg_item).flatten()



