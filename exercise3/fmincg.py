import scipy.optimize as op
import numpy as np
import costFunction

def fmincg(theta,x_data,y_data, lamda, num_labels):
    # 对每一类使用逻辑回归
    for i in range(num_labels):
        y_temp = y_data.copy()
        pos = np.where(y_data == i)
        neg = np.where(y_data != i)
        if i == 0:
            pos = np.where(y_data == 10)
            neg = np.where(y_data != 10)
        y_temp[pos] = 1
        y_temp[neg] = 0
        result = op.minimize(costFunction.cost,theta[i,:], args=(x_data,y_temp,lamda), method='TNC',jac=costFunction.gradient)
        print("{} : {}".format(i, result.success))
        theta[i, :] = result.x
    return theta
