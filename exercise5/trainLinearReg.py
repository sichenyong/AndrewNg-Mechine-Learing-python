import numpy as np
import scipy.optimize as op
import linearRegCostFunction
import matplotlib.pyplot as plt

def trainWithoutReg(theta,x_data, y_data, lamda):
    result = op.minimize(linearRegCostFunction.cost_reg,theta,args=(x_data,y_data,lamda),method="TNC",jac=linearRegCostFunction.gradient_reg)
    # print(result)
    return result.x, result.status



    