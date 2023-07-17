import numpy as np
import trainLinearReg
import linearRegCostFunction

def autoSelect(x_data, y_data, x_cv, y_cv ,lamdas):

    
    train_errors = np.zeros((len(lamdas),1))
    cv_errors = np.zeros((len(lamdas),1))

    for i in range(len(lamdas)):
        theta = np.zeros((x_data.shape[1] + 1,1))
        final_theta, _ = trainLinearReg.trainWithoutReg(theta,x_data,y_data,lamdas[i])
        train_errors[i, :] = linearRegCostFunction.cost_reg(final_theta, x_data, y_data,lamdas[i])
        cv_errors[i, :] = linearRegCostFunction.cost_reg(final_theta, x_cv, y_cv, lamdas[i])
    
    return train_errors, cv_errors
