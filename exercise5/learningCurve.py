import numpy as np
import matplotlib.pyplot as plt
import trainLinearReg
import linearRegCostFunction

def getError(x_train, y_train, x_cv, y_cv):
    m, n = x_train.shape
    errors_train = np.zeros((m, 1))
    errors_val = np.zeros((m, 1))

    for i in range(m):
        theta = np.zeros((n + 1,1))
        theta,_ = trainLinearReg.trainWithoutReg(theta,x_train[0:i+1, :], y_train[0:i+1, :], 0)
        errors_train[i, :] = linearRegCostFunction.cost_reg(theta,x_train[0:i+1,:],y_train[0:i+1,:],lamda=0)
        errors_val[i, :] = linearRegCostFunction.cost_reg(theta,x_cv,y_cv,lamda=0)
    return errors_train, errors_val