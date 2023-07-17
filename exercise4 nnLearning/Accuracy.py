import numpy as np
from scipy.special import expit

def cal_accuracy(x_data, y_data, theta1, theta2):
    m = x_data.shape[0]
    y_data = y_data.reshape((-1,1))
    correct = 0
    for i in range(m):
        label = y_data[i,:]
        #feedforward
        a1 = x_data[i,:].reshape((-1,1))
        label = y_data[i,:]
        # feedforward
        a1 = np.vstack((np.ones((1,1)),a1)) # add bais unit
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1,1)),a2)) 
        z3 = theta2 @ a2
        a3 = expit(z3) #hx
        index = np.argmax(a3)
        if index +1 == label:
            correct = correct + 1
    accuracy = correct / m
    return accuracy
