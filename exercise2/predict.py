import sigmoid
import numpy as np
def predict(theta, x_data):
    theta = theta.reshape((-1, 1))
    res = sigmoid.sigmoid(np.dot(x_data,theta))
    if res >= 0.5:
        return res,True
    else:
        return res, False 