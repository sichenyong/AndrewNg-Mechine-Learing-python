import numpy as np
import fmincg

def one_VS_all(x_data,y_data,lamda,num_labels):
    m,n = x_data.shape
    x_data = np.column_stack((np.ones((m,1)),x_data))
    initial_theta = np.zeros((num_labels, n + 1))
    final_theta = fmincg.fmincg(initial_theta,x_data,y_data,lamda,num_labels)
    return final_theta
