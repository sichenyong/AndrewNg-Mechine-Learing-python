import numpy as np
import costFunction

# 向前反馈
def predict(theta1, theta2,x_data):
    m,_ = x_data.shape
    a1 = x_data
    a1 = np.column_stack((np.ones((m, )), a1)) # add a0
    z2 = a1 @ theta1.T # z2 is 5000X25
    a2 = costFunction.sigmoid(z2)
    a2 = np.column_stack((np.ones(a2.shape[0],), a2)) # add a0
    z3 = a2 @ theta2.T # z3 is 5000 X 10
    a3 = costFunction.sigmoid(z3)
    hx = np.argmax(a3, axis=1) + 1
    return hx